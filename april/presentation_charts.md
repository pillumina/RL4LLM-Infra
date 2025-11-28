# VERL Partial Rollout - Charts and Diagrams

## 1. System Architecture Diagram

```mermaid
graph TB
    subgraph "VERL Rollout Worker"
        A[Input Prompts] --> B[PartialRolloutBuffer]
        B --> C[OversamplingController]
        C --> D[IndividualRequestGenerator]
        D --> E[SGLang Engine Interface]

        subgraph "Buffer Management"
            B --> F[Token Storage]
            B --> G[FIFO Eviction]
            B --> H[Step-based Cleanup]
        end
    end

    subgraph "SGLang Server"
        E --> I[Async Request Queue]
        I --> J[Tokenizer Manager]
        J --> K[GPU Workers]
        K --> L[Result Collector]
    end

    subgraph "Result Processing"
        L --> M[Finish Reason Parser]
        M --> N[Complete/Partial Classifier]
        N --> O[Buffer Storage]
        N --> P[Final Results]
        O --> B
    end

    style E fill:#e1f5fe
    style B fill:#f3e5f5
    style L fill:#e8f5e8
```

## 2. Request Lifecycle Flow Diagram

```mermaid
sequenceDiagram
    participant P as Prompts
    participant B as Buffer
    participant G as Generator
    participant S as SGLang
    participant A as AbortManager
    participant R as ResultProcessor

    P->>B: Check for continuation requests
    B-->>G: Return pending requests
    G->>G: Calculate oversampling size
    G->>S: Send individual requests (async)

    par Parallel Processing
        loop Until target reached
            S-->>R: Complete result
            R->>R: Check completion status
            alt Complete
                R->>A: Target reached?
                A->>S: Abort remaining requests
            end
        end
    and Partial Results
        S-->>R: Partial result
        R->>B: Store for continuation
    end

    R->>P: Return final results
```

## 3. Buffer State Transition Diagram

```mermaid
stateDiagram-v2
    [*] --> NewRequest: Submit request
    NewRequest --> InProgress: Start processing
    InProgress --> PartialComplete: Timeout/abort
    InProgress --> Complete: Normal finish
    PartialComplete --> Continuation: Retry next step
    Complete --> [*]: Return result
    Continuation --> InProgress: Process continuation

    state PartialComplete {
        [*] --> Stored
        Stored --> Evicted: Too old (step-based)
        Stored --> Retrieved: Next rollout step
        Evicted --> [*]
        Retrieved --> [*]
    }
```

## 4. Performance Comparison Chart

```mermaid
xychart-beta
    title "Performance Comparison: Traditional vs Partial Rollout"
    x-axis ["Training Speed", "GPU Utilization", "Memory Usage", "Implementation Complexity"]
    y-axis "Relative Performance (%)" 0 --> 150
    bar [100, 65, 100, 30]
    line [140, 90, 115, 85]
```

## 5. Oversampling Strategy Visualization

```mermaid
graph LR
    subgraph "Step N"
        A1[New Requests: 32] --> B1[Buffer: 16 pending]
        B1 --> C1[Target: 32]
        C1 --> D1[Send: 64 requests]
    end

    subgraph "SGLang Processing"
        D1 --> E1[40 Complete]
        D1 --> F1[24 Partial]
        E1 --> G1[Target reached ✓]
        F1 --> H1[Store to buffer]
    end

    subgraph "Step N+1"
        H1 --> A2[New Requests: 32]
        A2 --> B2[Buffer: 24 pending]
        B2 --> C2[Target: 32]
        C2 --> D2[Send: 64 requests]
    end

    style G1 fill:#c8e6c9
    style H1 fill:#fff3e0
```

## 6. Token Management Flow Diagram

```mermaid
flowchart TD
    A[Original Input Tokens] --> B[Generate Response]
    B --> C{Complete?}
    C -->|Yes| D[Return Full Result]
    C -->|No| E[Store Partial Result]
    E --> F[Next Rollout Step]
    F --> G[Concatenate: Original + Partial]
    G --> H[Generate Continuation]
    H --> I{Complete?}
    I -->|Yes| D
    I -->|No| E

    subgraph "Token Operations"
        J[Token ID: [10,20,30]]
        K[Partial: [100,101]]
        L[Continued: [10,20,30,100,101]]
        M[Response: [200,201,202]]
    end

    style J fill:#e3f2fd
    style K fill:#fce4ec
    style L fill:#f3e5f5
    style M fill:#e8f5e8
```

## 7. Error Handling Flow Diagram

```mermaid
flowchart TD
    A[Start Request] --> B[SGLang Call]
    B --> C{Success?}
    C -->|Yes| D[Process Result]
    C -->|No| E{Timeout?}
    E -->|Yes| F[Log Timeout Error]
    E -->|No| G{Network Error?}
    G -->|Yes| H[Log Network Error]
    G -->|No| I[Log Unknown Error]

    F --> J[Continue with other requests]
    H --> J
    I --> J

    D --> K{Target Reached?}
    K -->|Yes| L[Abort Remaining]
    K -->|No| M[Continue Processing]

    L --> N{Abort Success?}
    N -->|Yes| O[Collect Results]
    N -->|No| P[Log Abort Warning]
    P --> O

    M --> B
    J --> K
    O --> Q[Return Final Results]

    style F fill:#ffcdd2
    style H fill:#ffcdd2
    style I fill:#ffcdd2
    style P fill:#fff3e0
```

## 8. Memory Usage Timeline

```mermaid
gantt
    title Memory Usage Over Time
    dateFormat X
    axisFormat %s

    section Buffer Management
    Initial State :done, init, 0, 1s
    Store Partial Results :active, store, 1, 3s
    FIFO Eviction :evict, 2, 1s
    Step-based Cleanup :cleanup, 3, 1s

    section Request Processing
    Send Requests :send, 0, 1s
    Process Results :process, 1, 2s
    Collect Results :collect, 3, 1s

    section Memory Peaks
    Buffer Peak :milestone, peak1, 2s, 0s
    Processing Peak :milestone, peak2, 2s, 0s
```

## 9. Multi-Modal Data Flow

```mermaid
graph LR
    subgraph "Input Processing"
        A[Text Prompts] --> C[Tokenizer]
        B[Image Data] --> D[Image Processor]
        C --> E[Token IDs]
        D --> F[Image Embeddings]
    end

    subgraph "Buffer Storage"
        E --> G[Token Storage]
        F --> H[Multi-modal Storage]
        G --> I[Partial Buffer]
        H --> I
    end

    subgraph "Continuation"
        I --> J[Retrieve Data]
        J --> K[Reconstruct Input]
        K --> L[Send to SGLang]
        L --> M[Generate Response]
    end

    style A fill:#e3f2fd
    style B fill:#ffecb3
    style I fill:#f3e5f5
    style M fill:#e8f5e8
```

## 10. Configuration Impact Analysis

```mermaid
graph TD
    subgraph "Configuration Parameters"
        A[Oversampling Size] --> D[System Performance]
        B[Buffer Max Size] --> D
        C[Step Window] --> D
    end

    subgraph "Performance Metrics"
        D --> E[Throughput]
        D --> F[Latency]
        D --> G[Memory Usage]
        D --> H[Success Rate]
    end

    subgraph "Trade-offs"
        E --> I[Higher oversampling = Better throughput]
        F --> J[Smaller buffer = Lower latency]
        G --> K[Large buffer = More memory]
        H --> L[Optimal settings needed]
    end

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
```

## 11. Real-world Scenario Example

```mermaid
journey
    title Partial Rollout in Action
    section Request Submission
      Submit 32 prompts: 5: User
      Buffer has 12 pending: 3: System
      Calculate 2x oversampling: 3: System
      Send 64 requests: 5: User
    section Processing
      28 requests complete quickly: 3: System
      4 requests are slow: 1: User
      32 partial results saved: 3: System
      Target reached, abort remaining: 4: System
    section Results
      Collect 32 complete results: 5: User
      16 partial in buffer: 3: System
      Ready for next step: 5: User
```

## 12. Code Structure Overview

```mermaid
classDiagram
    class SGLangRollout {
        +config: RolloutConfig
        +_engine: SGLangEngine
        +partial_rollout_buffer: PartialRolloutBuffer
        +_batch_level_generate_sequences_with_partial_rollout()
        +_prepare_individual_requests_for_partial_rollout()
        +_execute_oversampled_requests_with_abort()
        +_send_individual_sglang_request()
        +_store_valuable_partial_results()
    }

    class PartialRolloutBuffer {
        +partial_requests: Dict
        +request_order: List
        +current_step: int
        +store_partial_requests()
        +get_continuation_requests()
        +remove_completed_requests()
        +increment_step()
        +get_stats()
    }

    class IndividualRequest {
        +request_id: str
        +input_ids: Tensor
        +sampling_params: Dict
        +is_continuation: bool
        +original_input_ids: Tensor
        +partial_response_token_ids: Tensor
        +remaining_max_tokens: int
    }

    class SGLangResponse {
        +meta_info: Dict
        +output_ids: List
        +logprob: List
        +finish_reason: str
    }

    SGLangRollout --> PartialRolloutBuffer
    SGLangRollout --> IndividualRequest
    SGLangRollout --> SGLangResponse
    PartialRolloutBuffer --> IndividualRequest
    IndividualRequest --> SGLangResponse
```

## 13. Debugging Flow Diagram

```mermaid
flowchart TD
    A[Problem Detected] --> B{Check Buffer State}
    B -->|Empty| C[Check Request Generation]
    B -->|Full| D[Check Eviction Logic]

    C --> E{Requests Generated?}
    E -->|No| F[Check Oversampling Config]
    E -->|Yes| G[Check SGLang Connection]

    D --> H{Eviction Working?}
    H -->|No| I[Check FIFO Logic]
    H -->|Yes| J[Check Step Window]

    G --> K{Connection OK?}
    K -->|No| L[Check SGLang Server]
    K -->|Yes| M[Check API Parameters]

    F --> N[Configuration Fix]
    I --> O[Code Fix]
    J --> P[Parameter Fix]
    L --> Q[Server Fix]
    M --> R[Parameter Fix]

    N --> S[Problem Resolved]
    O --> S
    P --> S
    Q --> S
    R --> S

    style S fill:#c8e6c9
    style A fill:#ffcdd2
```

## 14. Usage Guidelines

```mermaid
mindmap
  root((Partial Rollout))
    When to Use
      Long text generation
      Variable length outputs
      High throughput needed
      GPU resource constraints
    When to Avoid
      Fixed length requirements
      Synchronous processing needed
      Simple use cases
      Memory limited environments
    Best Practices
      Start with default config
      Monitor buffer statistics
      Adjust oversampling gradually
      Log abort operations
      Test with small batches first
    Common Pitfalls
      Too aggressive oversampling
      Buffer too large/small
      Ignoring error handling
      Not monitoring performance
```
