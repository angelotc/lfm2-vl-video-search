# Semantic Video Search - System Workflow

## High-Level Architecture

```mermaid
graph TB
    subgraph "Phase 1: Video Embedding Generation"
        V1[Upload Video] --> V2[Extract Frames]
        V2 --> V3[VLM Analysis<br/>Generate Descriptions]
        V3 --> V4[Text Embedding<br/>Sentence Transformers]
        V4 --> V5[Store JSON<br/>Text + Embeddings]
    end

    subgraph "Phase 2: Query & Retrieval"
        Q1[Enter Text Query] --> Q2[Embed Query<br/>Same Model]
        Q2 --> Q3[Search Stored Embeddings<br/>Cosine Similarity]
        Q3 --> Q4[Extract Video Clips<br/>±2s Padding]
        Q4 --> Q5[Display Results]
    end

    V5 --> Q3
    V1 -.-> Q1

    style V1 fill:#e1f5ff
    style V2 fill:#e1f5ff
    style V3 fill:#e1f5ff
    style V4 fill:#e1f5ff
    style V5 fill:#e1f5ff
    style Q1 fill:#ffe1e1
    style Q2 fill:#ffe1e1
    style Q3 fill:#ffe1e1
    style Q4 fill:#ffe1e1
    style Q5 fill:#ffe1e1
```

## Two-Phase Architecture Overview

```mermaid
flowchart LR
    subgraph "OFFLINE PHASE<br/>One-time per video"
        Video[Video File] --> Temporal[Temporal Analysis]
        Temporal --> Desc[Action Descriptions]
        Desc --> Embed[Text Embeddings]
        Embed --> Store[(Persistent Storage)]
    end

    subgraph "ONLINE PHASE<br/>Real-time queries"
        Query[User Query] --> QEmbed[Query Embedding]
        QEmbed --> Retrieve[Semantic Search]
        Store --> Retrieve
        Retrieve --> Clips[Clip Extraction]
        Clips --> Display[User Results]
    end

    Store -.->|Loaded at runtime| Retrieve

    style Video fill:#e1f5ff
    style Temporal fill:#e1f5ff
    style Desc fill:#e1f5ff
    style Embed fill:#e1f5ff
    style Store fill:#e1f5ff
    style Query fill:#ffe1e1
    style QEmbed fill:#ffe1e1
    style Retrieve fill:#ffe1e1
    style Clips fill:#ffe1e1
    style Display fill:#ffe1e1
```

## Component Architecture

```mermaid
graph TB
    subgraph "Embedding Generation Pipeline"
        Input[Video Input] --> Extract[Frame Extractor]
        Extract --> Triplets[Temporal Triplets<br/>N-1, N, N+1]
        Triplets --> VLM[VLM Model<br/>Action Detection]
        VLM --> TextGen[Text Generation]
        TextGen --> EmbedModel[Embedding Model<br/>Sentence Transformers]
        EmbedModel --> VectorStore[(Vector Store<br/>JSON File)]
    end

    subgraph "Query Pipeline"
        UserQuery[User Query] --> QueryEmbed[Query Embedding]
        QueryEmbed --> Similarity[Cosine Similarity<br/>Search Engine]
        VectorStore --> Similarity
        Similarity --> TopResults[Top-k Results]
        TopResults --> ClipGen[Clip Generation]
        ClipGen --> UserOutput[User Interface]
    end

    VectorStore -.->|Pre-loaded| Similarity

    style Input fill:#e1f5ff
    style VLM fill:#e1f5ff
    style EmbedModel fill:#e1f5ff
    style VectorStore fill:#e1f5ff
    style UserQuery fill:#ffe1e1
    style QueryEmbed fill:#ffe1e1
    style Similarity fill:#ffe1e1
    style ClipGen fill:#ffe1e1
```

## Data Flow Between Phases

```mermaid
flowchart TD
    subgraph "Phase 1: Offline Processing"
        VideoIn[Input Video<br/>.mp4/.mov] --> Process[Batch Processing<br/>5-15 sec/frame]
        Process --> Analysis[Temporal Analysis<br/>+15/-15 frame window]
        Analysis --> Description[Action Description<br/>"person jumping to shoot"]
        Description --> Vector[384-dim Vector<br/>Semantic Embedding]
        Vector --> Persist[(JSON Storage<br/>video_name.json)]
    end

    subgraph "Phase 2: Online Querying"
        TextInput[Text Query<br/>"layup"] --> Encode[Encode Query<br/>Same Model]
        Persist --> Load[Load Embeddings<br/>All Frames]
        Encode --> Compare[Compare Vectors<br/>Cosine Similarity]
        Load --> Compare
        Compare --> Rank[Rank Results<br/>By Similarity]
        Rank --> Extract[Extract Clips<br/>±2s padding]
        Extract --> Show[Display with Scores<br/>0.8745, 0.7231, ...]
    end

    Persist -.->|Shared Data| Load

    style VideoIn fill:#e1f5ff
    style Process fill:#e1f5ff
    style Analysis fill:#e1f5ff
    style Description fill:#e1f5ff
    style Vector fill:#e1f5ff
    style Persist fill:#e1f5ff
    style TextInput fill:#ffe1e1
    style Encode fill:#ffe1e1
    style Compare fill:#ffe1e1
    style Show fill:#ffe1e1
```

## Technology Stack

```mermaid
graph LR
    subgraph "Models & AI"
        VLM[LiquidAI LFM2-VL-450M<br/>3.2GB, Vision-Language]
        Embed[sentence-transformers<br/>90MB, Text Embeddings]
    end

    subgraph "Processing"
        OpenCV[OpenCV<br/>Video Processing]
        Streamlit[Streamlit<br/>Web UI]
        Python[Python<br/>Core Logic]
    end

    subgraph "Data"
        JSON[JSON Files<br/>Frame Analysis]
        Clips[MP4 Clips<br/>Extracted Segments]
        Embeddings[384-dim Vectors<br/>Semantic Search]
    end

    VLM --> Python
    Embed --> Python
    Python --> OpenCV
    Python --> Streamlit
    Python --> JSON
    Python --> Clips
    Embed --> Embeddings

    style VLM fill:#ffe1e1
    style Embed fill:#ffe1e1
    style JSON fill:#e1ffe1
    style Clips fill:#fff4e1
    style Embeddings fill:#e1ffe1
```

## Performance & Scalability

```mermaid
flowchart TD
    subgraph "Offline Generation"
        B1[1 Video] --> B2[30 min video]
        B2 --> B3[~60 frames]
        B3 --> B4[~5-15 min processing]
        B4 --> B5[~1.5MB JSON<br/>+ embeddings]
    end

    subgraph "Online Querying"
        Q1[1 Query] --> Q2[~100ms embedding]
        Q2 --> Q3[~10ms search<br/>across 60 frames]
        Q3 --> Q4[~500ms clip extraction]
        Q4 --> Q5[~600ms total]
    end

    subgraph "Scalability Limits"
        S1[CPU only<br/>No GPU acceleration]
        S2[Memory intensive<br/>All frames loaded]
        S3[JSON storage<br/>No vector DB]
        S4[Single video<br/>per search]
    end

    style B4 fill:#fff4e1
    style Q5 fill:#e1ffe1
    style S1 fill:#ffcccc
    style S2 fill:#ffcccc
    style S3 fill:#ffcccc
    style S4 fill:#ffcccc
```

## Complete System Diagram

```mermaid
flowchart TD
    Start([User runs streamlit run main.py]) --> Upload[Upload Video File]
    Upload --> CheckJSON{JSON exists for<br/>this video?}

    CheckJSON -->|Yes| LoadJSON[Load existing JSON analysis]
    CheckJSON -->|No| Analyze[Start Video Analysis]

    LoadJSON --> ShowResults[Display frame count & results]
    ShowResults --> SearchUI[Show Search Interface]

    Analyze --> LoadModels[Load VLM & Embedding Models]
    LoadModels --> ExtractFrames[Extract All Frames to Memory]
    ExtractFrames --> CreateTriplets[Create Temporal Frame Triplets<br/>Before/Current/After]
    CreateTriplets --> BatchProcess[Process Batches Sequentially]

    BatchProcess --> VLMDesc[VLM generates action descriptions<br/>for each frame triplet]
    VLMDesc --> EmbedText[Embed descriptions using<br/>sentence-transformers]
    EmbedText --> SaveIncremental{Processed 8<br/>frames?}

    SaveIncremental -->|Yes| SaveJSON[Save checkpoint JSON<br/>status: in_progress]
    SaveIncremental -->|No| MoreBatches{More<br/>batches?}
    SaveJSON --> MoreBatches

    MoreBatches -->|Yes| BatchProcess
    MoreBatches -->|No| SaveFinal[Save final JSON<br/>status: complete]
    SaveFinal --> SearchUI

    SearchUI --> QueryInput[Enter search query<br/>e.g., 'layup']
    QueryInput --> EmbedQuery[Embed query text using<br/>same embedding model]
    EmbedQuery --> CosineSim[Calculate cosine similarity<br/>between query & all frame embeddings]
    CosineSim --> TopK[Return top-k results<br/>sorted by similarity]

    TopK --> DisplayResults[For each result...]
    DisplayResults --> CalcTimestamp[Calculate frame timestamp<br/>from frame number & FPS]
    CalcTimestamp --> ExtractClip[Extract video clip with<br/>±2s padding]
    ExtractClip --> ShowClip[Display clip & similarity score]
    ShowClip --> Download[Download button for clip]
    Download --> NextResult{More<br/>results?}

    NextResult -->|Yes| DisplayResults
    NextResult -->|No| Done([User can query again or<br/>re-analyze video])

    style Start fill:#e1f5ff
    style Done fill:#e1f5ff
    style SearchUI fill:#ffe1e1
    style SaveFinal fill:#e1ffe1
    style LoadJSON fill:#fff4e1
```

## Data Flow Diagram

```mermaid
flowchart LR
    Video[(Video File<br/>wisconsin-vs-montana.mp4)] --> FrameExtract[Frame Extraction]
    FrameExtract --> Frames[(All Video Frames<br/>in Memory)]

    Frames --> Sample[Sample every 30 frames]
    Sample --> Triplets[(Frame Triplets<br/>N-1, N, N+1)]

    Triplets --> VLM[VLM Model<br/>LFM2-VL-450M]
    VLM --> Descriptions[(Text Descriptions<br/>person jumping to shoot)]

    Descriptions --> Embedder[Sentence Transformer<br/>all-MiniLM-L6-v2]
    Embedder --> Embeddings[(384-dim Vectors)]

    Descriptions --> JSON[(JSON File<br/>wisconsin-vs-montana.json)]
    Embeddings --> JSON

    JSON --> Search[Search Function]
    Query[(User Query<br/>layup)] --> QueryEmbed[Embed Query]
    QueryEmbed --> Search

    Search --> Similarity[Cosine Similarity]
    Similarity --> Results[(Top-k Results<br/>with scores)]

    Results --> ClipExtract[Clip Extractor]
    Video --> ClipExtract
    ClipExtract --> Clips[(Video Clips<br/>±2s padding)]

    style Video fill:#e1f5ff
    style JSON fill:#e1ffe1
    style Query fill:#ffe1e1
    style Clips fill:#fff4e1
```

## Architecture Diagram

```mermaid
graph TB
    subgraph "Frontend - Streamlit UI"
        UI[main.py]
        Upload[File Uploader]
        SearchBox[Search Query Input]
        Results[Results Display]
        VideoPlayer[Video Clip Player]
    end

    subgraph "Processing Layer"
        JSONCheck[JSON Detection]
        FrameProc[Frame Processor]
        VLM[VLM Model]
        Embedder[Embedding Model]
        SearchEngine[Search Engine]
        ClipExtractor[Clip Extractor]
    end

    subgraph "Storage Layer"
        VideoStore[(uploaded video<br/>extracted_clips/)]
        JSONStore[(frame analysis<br/>video_frames_analysis/)]
    end

    subgraph "Models - Cached"
        VLMModel[LiquidAI/LFM2-VL-450M<br/>~3.2GB]
        EmbedModel[sentence-transformers<br/>all-MiniLM-L6-v2<br/>~90MB]
    end

    UI --> Upload
    Upload --> JSONCheck
    JSONCheck --> JSONStore

    JSONCheck -->|No JSON| FrameProc
    FrameProc --> VLM
    VLM --> VLMModel
    VLM --> Embedder
    Embedder --> EmbedModel
    Embedder --> JSONStore

    JSONCheck -->|JSON exists| SearchBox
    SearchBox --> SearchEngine
    SearchEngine --> EmbedModel
    SearchEngine --> JSONStore
    SearchEngine --> Results

    Results --> ClipExtractor
    ClipExtractor --> VideoStore
    ClipExtractor --> VideoPlayer

    style UI fill:#e1f5ff
    style VLMModel fill:#ffe1e1
    style EmbedModel fill:#ffe1e1
    style JSONStore fill:#e1ffe1
```

## Sequence Diagram - Search Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant FileCheck as JSON Checker
    participant Storage as video_frames_analysis/
    participant Embedder as Embedding Model
    participant Search as Search Engine
    participant Clip as Clip Extractor

    User->>UI: Upload wisconsin-vs-montana.mp4
    UI->>FileCheck: Check for JSON
    FileCheck->>Storage: Look for wisconsin-vs-montana.json
    Storage-->>FileCheck: ✓ JSON found
    FileCheck-->>UI: Load existing analysis

    UI->>User: Show search interface
    User->>UI: Enter query "layup"
    UI->>Embedder: Embed query text
    Embedder-->>UI: Query embedding [384-dim]

    UI->>Search: Find similar frames
    Search->>Storage: Load all frame embeddings
    Storage-->>Search: Frame embeddings + descriptions
    Search->>Search: Calculate cosine similarity
    Search-->>UI: Top-5 results with scores

    loop For each result
        UI->>Clip: Extract clip (timestamp ± 2s)
        Clip->>Clip: Read video frames
        Clip-->>UI: Video clip file
        UI->>User: Display clip + score + download
    end

    User->>UI: Click download
    UI-->>User: Save clip_{i}_frame{n}_layup.mp4
```

## File Structure Diagram

```mermaid
graph TD
    Root[lf2-vl-video-search/] --> App[app.py<br/>Basic upload UI]
    Root --> Main[main.py<br/>Main application<br/>3200+ lines]
    Root --> Search[search.py<br/>CLI search tool]
    Root --> Claude[CLAUDE.md<br/>Documentation]
    Root --> Req[requirements.txt]

    Root --> UploadsDir[uploads/]
    Root --> AnalysisDir[video_frames_analysis/]
    Root --> ClipsDir[extracted_clips/]

    AnalysisDir --> JSON1[wisconsin-vs-montana-clip.json]
    AnalysisDir --> JSON2[video_analysis1.json]

    ClipsDir --> Clip1[clip_1_frame15_layup.mp4]
    ClipsDir --> Clip2[clip_2_frame28_layup.mp4]
    ClipsDir --> TempVid[temp_wisconsin-vs-montana-clip.mp4]

    JSON1 --> Structure{JSON Structure}
    Structure --> VidName[video_name: str]
    Structure --> TotalFrames[total_frames: int]
    Structure --> Status[status: complete/in_progress]
    Structure --> FramesList[frames: Array]

    FramesList --> Frame{Frame Object}
    Frame --> FrameNum[frame_number: int]
    Frame --> Text[text: str<br/>action description]
    Frame --> Embedding[embedding: float[]<br/>384 dimensions]

    style Main fill:#e1f5ff
    style AnalysisDir fill:#e1ffe1
    style ClipsDir fill:#fff4e1
    style JSON1 fill:#ffe1e1
```

## Model Processing Pipeline

```mermaid
flowchart LR
    subgraph Input
        V[Video Frame N-1]
        C[Video Frame N]
        A[Video Frame N+1]
    end

    subgraph "VLM Processing"
        Conv[Multi-image<br/>Conversation Template]
        VLM[LiquidAI LFM2-VL-450M<br/>Image-Text-to-Text]
        Gen[Generate Description<br/>max 128 tokens]
    end

    subgraph "Embedding Generation"
        Desc[Text Description<br/>person jumping to shoot]
        Embed[Sentence Transformer<br/>all-MiniLM-L6-v2]
        Vec[384-dim Vector<br/>[0.123, -0.456, ...]]
    end

    subgraph Output
        JSON[JSON Entry:<br/>frame_number, text, embedding]
    end

    V --> Conv
    C --> Conv
    A --> Conv
    Conv --> VLM
    VLM --> Gen
    Gen --> Desc
    Desc --> Embed
    Embed --> Vec

    Desc --> JSON
    Vec --> JSON

    style Input fill:#e1f5ff
    style VLM fill:#ffe1e1
    style Embed fill:#ffe1e1
    style Output fill:#e1ffe1
```

