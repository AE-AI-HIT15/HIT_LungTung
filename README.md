
HIT Product 2025
# GenImageAI
A system that generates high-quality images from natural language text prompts

## 📋 Overview

This project implements a full-stack application featuring:

- A FastAPI backend serving the text-to-image model for inference
- An HTML/CSS/JavaScript frontend for interactive prompt input and image display
- Integration with a translation module to support multilingual prompts
- GPU acceleration for fast image synthesis using Stable Diffusion XL
- Docker for containerization and easy deployment

## 🏛️ System Architecture

```mermaid
graph TB
    subgraph "User Layer"
        User["👤 User"]
        Browser["🌐 Web Browser<br/>(Port 8080)"]
    end

    subgraph "Frontend Layer"
        UI["📱 Interactive UI<br/>(index.html)"]
        JS["⚡ JavaScript Client<br/>- Event Handling<br/>- API Communication<br/>- Dynamic Rendering"]
    end

    subgraph "Backend Layer - FastAPI Application"
        API["🔌 REST API Gateway<br/>(Port 8000)<br/>main.py"]
        Router["🔀 Text2Image Router<br/>/v1/Text2Image<br/>api/routers/text2image.py"]
        Handler["⚙️ Exception Handler<br/>api/helpers/exception_handler.py"]
    end

    subgraph "Application Layer"
        LangDetect["🌍 Language Detector<br/>langdetect"]
        Translator["🔤 Vietnamese → English<br/>GoogleTranslator<br/>deep_translator"]
        T2IService["🎨 Text2Image Service<br/>infra/load_model/load_model.py"]
    end

    subgraph "Model Layer - GPU Accelerated"
        SDXL["🤖 Stable Diffusion XL Pipeline<br/>StableDiffusionXLPipeline"]
        LoRA["✨ LoRA Weights<br/>(Optional Fine-tuning)<br/>backend/shared/weights"]
        GPU["🚀 CUDA GPU<br/>FP16 Precision<br/>torch.float16"]
    end

    subgraph "Infrastructure Layer"
        Settings["⚙️ Configuration<br/>shared/settings/<br/>.env"]
        Logger["📝 Logging System<br/>shared/logging/"]
        BaseModels["📦 Base Models<br/>shared/base/"]
    end

    subgraph "Output Layer"
        ImageGen["🖼️ Generated Image<br/>(PIL Image)"]
        Base64["🔐 Base64 Encoder<br/>PNG Format"]
        Response["📤 JSON Response<br/>{image_base64: '...'}"]
    end

    %% User Flow
    User -->|"1. Enters Prompt"| Browser
    Browser -->|"2. HTTP Request"| UI
    UI <-->|"3. Events & DOM"| JS
    
    %% API Communication
    JS -->|"4. POST /v1/Text2Image<br/>JSON: {prompt: '...'}"| API
    API -->|"5. Route Request"| Router
    Router -->|"6. Validate Input"| Handler
    
    %% Processing Flow
    Router -->|"7. Process Request"| T2IService
    T2IService -->|"8. Detect Language"| LangDetect
    LangDetect -->|"9. Vietnamese?"| Translator
    Translator -->|"10. Translated Prompt"| T2IService
    
    %% Model Inference
    T2IService -->|"11. Load Pipeline<br/>(cached_property)"| SDXL
    Settings -.->|"Model Config"| SDXL
    LoRA -.->|"Fine-tune Weights"| SDXL
    SDXL -->|"12. Run on"| GPU
    GPU -->|"13. Inference Result"| ImageGen
    
    %% Response Flow
    ImageGen -->|"14. Encode"| Base64
    Base64 -->|"15. Create Output"| Response
    Response -->|"16. JSON Response"| Router
    Router -->|"17. HTTP 200 OK"| JS
    JS -->|"18. Display Image"| UI
    UI -->|"19. Visual Output"| Browser
    Browser -->|"20. View Image"| User
    
    %% Infrastructure Support
    Logger -.->|"Monitor"| Router
    Logger -.->|"Monitor"| T2IService
    BaseModels -.->|"Base Classes"| T2IService
    
    style User fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    style Browser fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style UI fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style API fill:#e8f5e9,stroke:#1b5e20,stroke-width:3px
    style SDXL fill:#fff9c4,stroke:#f57f17,stroke-width:3px
    style GPU fill:#ffebee,stroke:#b71c1c,stroke-width:3px
    style ImageGen fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    style Translator fill:#fce4ec,stroke:#880e4f,stroke-width:2px
```

### 🔄 Data Flow Summary

1. **User Input**: User enters a text prompt (Vietnamese or English) in the web interface
2. **Frontend Processing**: JavaScript captures and sends the prompt via POST request
3. **API Gateway**: FastAPI receives and routes the request to Text2Image endpoint
4. **Language Processing**: 
   - Detects if prompt is in Vietnamese
   - Translates to English if needed (improves generation quality)
5. **Model Inference**: 
   - Loads cached Stable Diffusion XL pipeline with LoRA weights
   - Runs inference on GPU with FP16 precision
6. **Image Encoding**: Converts PIL image to base64-encoded PNG
7. **Response Delivery**: Returns JSON with base64 image to frontend
8. **Display**: JavaScript decodes and displays the generated image

## 🎓 Training Architecture

The project includes an SDXL LoRA training script ([`train_text_to_image_lora_sdxl.py`](file:///e:/HIT_LungTung-1/train_text_to_image_lora_sdxl.py)) for fine-tuning custom models:

```mermaid
graph TB
    subgraph "Input"
        Dataset["📊 Dataset<br/>Images + Captions<br/>metadata.jsonl"]
        Config["⚙️ Config<br/>Hyperparameters<br/>Model Paths"]
    end

    subgraph "Model Initialization"
        BaseModel["🤖 SDXL Base Model<br/>UNet + VAE + Text Encoders"]
        LoRA["✨ LoRA Adapters<br/>Rank 4-8<br/>Attention Layers Only"]
    end

    subgraph "Training Pipeline"
        DataPrep["🖼️ Data Preparation<br/>Resize to 1024x1024<br/>Dual Tokenization"]
        TrainLoop["🔄 Training Loop<br/>VAE Encode → Add Noise<br/>UNet Predict → MSE Loss<br/>Backprop LoRA params"]
        Validation["✅ Validation<br/>Generate Images<br/>Log Metrics"]
    end

    subgraph "Output"
        Checkpoint["💾 Checkpoints<br/>Every 500 steps"]
        LoRAWeights["💎 Final LoRA Weights<br/>*.safetensors"]
    end

    Dataset --> DataPrep
    Config --> BaseModel
    BaseModel --> LoRA
    LoRA --> TrainLoop
    DataPrep --> TrainLoop
    TrainLoop --> Validation
    TrainLoop --> Checkpoint
    Validation --> LoRAWeights
    Checkpoint --> LoRAWeights

    style Dataset fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style LoRA fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style TrainLoop fill:#e8f5e9,stroke:#388e3c,stroke-width:3px
    style LoRAWeights fill:#fff9c4,stroke:#f57f17,stroke-width:3px
```

**Key Features:**
- **Parameter-Efficient**: Only trains LoRA adapters (~1-5% of model params)
- **Mixed Precision**: FP16/BF16 for memory efficiency
- **Distributed Training**: Multi-GPU support via Accelerate
- **Flexible**: Custom datasets, hyperparameters, validation prompts

**Quick Start:**
```bash
python train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="./data/training" \
  --output_dir="./output/lora-model" \
  --resolution=1024 \
  --train_batch_size=4 \
  --learning_rate=1e-4 \
  --rank=8 \
  --mixed_precision="fp16"
```

## ⚙️ Components

### Backend

- **FastAPI Application**: REST API with structured routes
- **Text2Image Service**: Implements text-to-image generation using the Stable Diffusion XL pipeline with optional LoRA weights for fine-tuning
- **Prompt Language Detection and Translation**: Automatically detects if the input prompt is in Vietnamese and translates it to English to improve generation quality
- **Model Management**: Loads and caches the Stable Diffusion XL model with FP16 precision on GPU for efficient inference.
- **Image Encoding**: Converts generated images to base64-encoded PNG format for easy API response handling.

### Frontend

- **Static index.html**: The user interface is built with a simple static index.html file for ease of use and deployment.

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- Hugging Face Transformers and Diffusers libraries
- safetensors library for loading LoRA weights
- Pretrained SDXL base model

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd HIT_LungTung
   ```

2. Set up environment variables:
   ```bash
   cp .env.template .env
   ```

   Then, edit the `.env` file to configure your personal settings with the following content:
   ```
   T2I__BASE_MODEL_ID='YOUR_BASE_MODEL_ID'
   T2I__LORA_WEIGHTS='shared/weights'

   ```
3. Setup Model Files
- Create a folder named `weights` inside the folder: backend\shared
- Download the model file from this Google Drive link: https://drive.google.com/file/d/1xrItYfDMEhbO3C2FTOrXF6e0nVHQleLc/view?usp=drive_link

3. Start the application using docker compose:
    ```
    docker compose up --build
    ```
## 🏗️ Project Structure
```
├── backend                         # Backend FastAPI application
│   ├── api/                        # Contains API-related files, e.g., routes, API helpers
│   ├── application/                # Core application logic and service layer
│   ├── domain/                     # Entities and core business logic of the system
│   ├── infra/                      # Infrastructure components, e.g., DB connection, external services, system config
│   ├── shared/                     # Modules, functions, configs shared across the project
│   ├── test/                       # Test files (unit tests, integration tests)
│   ├── Dockerfile                  # Docker configuration file to build backend container
│   ├── main.py                     # Main entry point of backend application (e.g., FastAPI app initialization)
│   └── requirements.txt            # List of Python dependencies to install
├── frontend                        # Frontend application
│   ├── __init__.py                 # Package initializer for frontend (can be empty)
│   ├── index.html                  # Main web interface file of the frontend
│   └── Logo.png                    # Logo image used in frontend or displayed on UI
├── .dockerignore                   # Docker ignore file
├── docker-compose.yml              # Docker Compose config for orchestrating services
├── pyproject.toml                  # Python project configuration file
├── train_text_to_image_lora_sdxl.py# SDXL LoRA training script for model fine-tuning
└── README.md                       # Project documentation

```

## 💻 Usage

Once the application is running:

1. Access the frontend at: http://localhost:8080
2. The backend API is available at: http://localhost:8000


