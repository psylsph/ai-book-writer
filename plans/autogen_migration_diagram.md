# AutoGen Migration Architecture Diagram

## Current Architecture (Legacy API)

```mermaid
graph TB
    subgraph "Legacy AutoGen Architecture"
        A[agents.py] -->|Creates| B[ConversableAgent]
        B -->|Uses| C[llm_config dict]
        
        D[agent_factory.py] -->|Creates| E[GroupChat]
        E -->|Managed by| F[GroupChatManager]
        
        G[book_generator.py] -->|Uses| F
        G -->|Calls| H[initiate_chat]
        
        I[outline_generator.py] -->|Uses| F
        I -->|Calls| H
    end
    
    style B fill:#ffcccc
    style E fill:#ffcccc
    style F fill:#ffcccc
```

## Target Architecture (AutoGen 2.0)

```mermaid
graph TB
    subgraph "AutoGen 2.0 Architecture"
        A[agents.py] -->|Creates| B[AssistantAgent]
        B -->|Uses| C[model_client object]
        
        D[agent_factory.py] -->|Creates| E[RoundRobinGroupChat Team]
        E -->|Directly runs| F[team.run]
        
        G[book_generator.py] -->|Uses| E
        G -->|Await| H[team.run_stream]
        
        I[outline_generator.py] -->|Uses| E
        I -->|Await| H
    end
    
    style B fill:#ccffcc
    style E fill:#ccffcc
    style H fill:#ccffcc
```

## Migration Flow

```mermaid
graph LR
    A[Phase 1: Config] --> B[Phase 2: Agents]
    B --> C[Phase 3: Factory]
    C --> D[Phase 4: Book Generator]
    D --> E[Phase 5: Outline Generator]
    E --> F[Phase 6: Testing]
    F --> G[Phase 7: Documentation]
    
    style A fill:#e1f5ff
    style B fill:#e1f5ff
    style C fill:#fff4e1
    style D fill:#fff4e1
    style E fill:#fff4e1
    style F fill:#e8f5e9
    style G fill:#e8f5e9
```

## Key Component Mappings

| Legacy Component | New Component | Notes |
|-----------------|---------------|-------|
| `ConversableAgent` | `AssistantAgent` | Different initialization |
| `llm_config` dict | `model_client` object | More explicit configuration |
| `GroupChat` | `RoundRobinGroupChat` | Team-based approach |
| `GroupChatManager` | `team.run()` | Direct execution, no manager |
| `initiate_chat()` | `await team.run()` | Async by default |
| `messages` list | `result.messages` | Different message object type |
| `speaker_selection_method` | Team type | Built into team selection |

## File Dependency Graph

```mermaid
graph TD
    A[config.py] -->|Provides| B[model_client]
    
    C[agents.py] -->|Uses| B
    C -->|Creates| D[Agents]
    
    E[agent_factory.py] -->|Uses| D
    E -->|Creates| F[Teams]
    
    G[book_generator.py] -->|Uses| D
    G -->|Uses| F
    
    H[outline_generator.py] -->|Uses| D
    H -->|Uses| F
    
    style A fill:#ffe1e1
    style B fill:#ffe1e1
    style C fill:#e1f5ff
    style D fill:#e1f5ff
    style E fill:#fff4e1
    style F fill:#fff4e1
    style G fill:#e8f5e9
    style H fill:#e8f5e9
```

## Migration Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Async conversion complexity | High | Incremental async migration, wrappers |
| Message format changes | Medium | Adapter layer for message conversion |
| Breaking changes in team API | High | Thorough testing at each phase |
| Performance differences | Medium | Benchmark before/after |
| Documentation gaps | Low | Document as we migrate |

## Testing Strategy

```mermaid
graph TD
    A[Unit Tests] -->|Validate| B[Agent Creation]
    A -->|Validate| C[Team Creation]
    
    D[Integration Tests] -->|Validate| E[Outline Generation]
    D -->|Validate| F[Chapter Generation]
    
    G[E2E Tests] -->|Validate| H[Full Book Generation]
    
    I[Performance Tests] -->|Validate| J[Response Times]
    I -->|Validate| K[Memory Usage]
    
    style A fill:#e1f5ff
    style D fill:#fff4e1
    style G fill:#e8f5e9
    style I fill:#fce4ec
```

## Rollback Strategy

```mermaid
graph LR
    A[Migration Issue Detected] --> B{Severity?}
    B -->|Critical| C[Immediate Rollback]
    B -->|Minor| D[Fix & Continue]
    
    C --> E[Git Revert]
    C --> F[Reinstall Legacy]
    C --> G[Restore Tests]
    
    D --> H[Fix Issue]
    H --> I[Continue Migration]
    
    style C fill:#ffcccc
    style E fill:#ffcccc
    style F fill:#ffcccc
    style G fill:#ffcccc
```
