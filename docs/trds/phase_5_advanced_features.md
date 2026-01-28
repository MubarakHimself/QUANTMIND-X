# Phase 5: Advanced Features

> **Focus:** Context management, modes, external integrations, polish
> **Dependencies:** Phase 4 complete (Kanban and orchestration working)
> **Deliverable:** Production-ready backend with all features

---

## Overview

This phase adds all the advanced features: mode system, context offloading, intent journal, model router, notifications, and voice interface (STT + TTS).

**Note:** Open Deep Research integration has been removed from Phase 5 scope to prioritize voice interface capabilities.

---

## Sub-Phases

### 5.1 Mode System (XML Templates)

**Goal:** Different operational modes with distinct system prompts

**Tasks:**
1. Mode definitions:
```python
class Mode(Enum):
    CHAT = "chat"
    UPGRADE = "upgrade"
    DEPARTMENT_CREATION = "department_creation"
    SKILL_CREATION = "skill_creation"
    MCP_CREATION = "mcp_creation"
    BRAINSTORMING = "brainstorming"
```

2. XML template files:
```
/templates/modes/
├── chat.xml
├── upgrade.xml
├── department_creation.xml
├── skill_creation.xml
├── mcp_creation.xml
└── brainstorming.xml
```

3. Template format:
```xml
<!-- chat.xml -->
<system_prompt>
  <identity name="roi" version="1.0">
    You are ROI (Rigged Oriented Intelligence), a personal AI assistant.
  </identity>
  
  <tools enabled="{tools_enabled}">
    <tool name="filesystem" active="{enable_filesystem}"/>
    <tool name="kanban" active="true"/>
    <tool name="memory" active="true"/>
  </tools>
  
  <context>
    <mode>chat</mode>
    <user_preferences>{user_preferences}</user_preferences>
  </context>
  
  <rules>
    <rule>Always be helpful and conversational</rule>
    <rule>Remember user preferences</rule>
  </rules>
</system_prompt>
```

4. Mode switcher:
```python
class ModeManager:
    def __init__(self):
        self.current_mode = Mode.CHAT
        self.templates = self._load_templates()
    
    def switch_mode(self, mode: Mode, context: dict) -> str:
        """Switch mode and return compiled system prompt."""
        template = self.templates[mode]
        return self._compile_template(template, context)
    
    def _compile_template(self, template: str, context: dict) -> str:
        """Replace {variables} with values."""
        for key, value in context.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template
```

5. Mode-specific tool sets

**Success Criteria:**
- [ ] All mode templates created
- [ ] Mode switching working
- [ ] Tools enable/disable per mode
- [ ] Context injected correctly

**Test:**
```python
prompt = mode_manager.switch_mode(Mode.BRAINSTORMING, {"tools_enabled": "false"})
assert "<tool name=\"filesystem\" active=\"false\"/>" in prompt
```

---

### 5.2 Context Offloading (Chat Tree)

**Goal:** Automatic context management at 40% threshold

**Tasks:**
1. Context tracker:
```python
class ContextTracker:
    def __init__(self, model_context_limit: int):
        self.limit = model_context_limit
        self.tokens_used = 0
        self.threshold = 0.4
    
    def add_tokens(self, count: int):
        self.tokens_used += count
    
    def should_offload(self) -> bool:
        return (self.tokens_used / self.limit) >= self.threshold
    
    def reset(self):
        self.tokens_used = 0
```

2. Chat tree manager:
```python
class ChatTreeManager:
    def __init__(self, session_id: str, tracker: ContextTracker):
        self.session_id = session_id
        self.tracker = tracker
        self.current_node = 0
    
    async def check_and_offload(self, messages: list) -> list:
        if self.tracker.should_offload():
            # 1. Save current node to LangMem
            await self._save_node(messages)
            
            # 2. Extract key points for new context
            summary = await self._summarize(messages)
            
            # 3. Start new node
            self.current_node += 1
            self.tracker.reset()
            
            # 4. Return fresh context with summary
            return [{"role": "system", "content": f"Previous context summary: {summary}"}]
        
        return messages
    
    async def _save_node(self, messages: list):
        """Save to episodic memory."""
        # Use LangMem to store
        
    async def _summarize(self, messages: list) -> str:
        """Create concise summary of conversation so far."""
```

3. Integration with chat endpoint:
```python
@router.post("/chat")
async def chat(message: ChatMessage):
    # Track incoming message tokens
    tracker.add_tokens(count_tokens(message.content))
    
    # Check for offload
    messages = await chat_tree.check_and_offload(messages)
    
    # Continue conversation
    response = await agent.invoke(messages)
    
    # Track response tokens
    tracker.add_tokens(count_tokens(response))
    
    return response
```

4. UI representation (for future):
   - Chat appears continuous
   - Internal node boundaries invisible
   - User can view chat tree structure if desired

**Success Criteria:**
- [ ] Token counting accurate
- [ ] Offload triggers at 40%
- [ ] Context summarized
- [ ] Conversation continues seamlessly

**Test:**
1. Send many messages until 40% reached
2. Verify offload triggered
3. Continue conversation
4. Verify memory recall works

---

### 5.3 Intent Journal

**Goal:** Track user intents and preferences over time

**Tasks:**
1. Intent schema:
```python
class Intent(BaseModel):
    date: str
    category: str  # preference, decision, goal
    content: str
    source: str  # conversation_id
```

2. Intent extractor:
```python
class IntentExtractor:
    def __init__(self, llm):
        self.llm = llm
    
    async def extract_intents(self, messages: list) -> list[Intent]:
        """Extract intents from conversation."""
        prompt = """
        Analyze this conversation and extract:
        1. User preferences (e.g., "prefers markdown", "likes detailed reports")
        2. Decisions made (e.g., "chose PostgreSQL over SQLite")
        3. Goals stated (e.g., "wants to automate research")
        
        Return as JSON list.
        """
        # ...
```

3. Intent storage (markdown):
```markdown
# Intent Journal

## 2026-01-12

### Preferences
- User prefers markdown logs over JSON
- User wants Docker sandbox + venv toggle

### Decisions
- ROI should use Piper TTS for voice
- No complex coding in ROI (use OpenCode)

### Goals
- Build a Claude Desktop-like personal assistant
- Department-based AI employees
```

4. Intent injection at conversation start:
```python
def get_recent_intents(limit: int = 10) -> str:
    """Get recent intents for context injection."""
    intents = db.query(Intent).order_by(Intent.date.desc()).limit(limit)
    return format_intents(intents)
```

**Success Criteria:**
- [ ] Intents extracted from conversations
- [ ] Saved to markdown + database
- [ ] Injected at conversation start
- [ ] Useful for ROI context

**Test:**
1. Have conversation with stated preferences
2. Verify intents extracted
3. Start new conversation
4. Verify intents in context

---

### 5.4 Model Router (Multi-Model Selection)

**Goal:** Select from local and API models

**Tasks:**
1. Model registry:
```python
class ModelConfig(BaseModel):
    name: str
    provider: str  # ollama, anthropic, openai, google
    model_id: str
    is_local: bool
    context_limit: int
    capabilities: list[str]  # text, vision, code

models = {
    "claude-sonnet": ModelConfig(
        name="Claude Sonnet",
        provider="anthropic",
        model_id="claude-3-5-sonnet",
        is_local=False,
        context_limit=200000,
        capabilities=["text", "code"]
    ),
    "llama3.2": ModelConfig(
        name="Llama 3.2",
        provider="ollama",
        model_id="llama3.2",
        is_local=True,
        context_limit=128000,
        capabilities=["text"]
    ),
    # ...
}
```

2. Model selection API:
```python
@router.get("/models")
async def list_models():
    """List available models."""
    return list(models.values())

@router.post("/models/select")
async def select_model(model_name: str):
    """Select model for current session."""
    session.current_model = models[model_name]
```

3. Dynamic model initialization:
```python
def get_model(model_name: str):
    config = models[model_name]
    
    match config.provider:
        case "ollama":
            return ChatOllama(model=config.model_id)
        case "anthropic":
            return ChatAnthropic(model=config.model_id)
        case "openai":
            return ChatOpenAI(model=config.model_id)
        case "google":
            return ChatGoogleGenerativeAI(model=config.model_id)
```

4. Per-Kanban-card model selection

**Success Criteria:**
- [ ] Multiple models registered
- [ ] Can switch models via API
- [ ] Local and API models work
- [ ] Per-card model selection

**Test:**
```bash
curl http://localhost:8000/models
# Returns list of available models

curl -X POST http://localhost:8000/models/select -d '{"model_name": "llama3.2"}'
# Switches to local model
```

---

### 5.5 Open Deep Research Integration

**⚠️ REMOVED FROM PHASE 5 SCOPE**

This feature has been removed from Phase 5 to prioritize voice interface capabilities. May be reconsidered for future phases.

---

### 5.6 Notification System

**Goal:** Desktop + WhatsApp notifications

**Tasks:**
1. Notification service:
```python
class NotificationService:
    def __init__(self, config: NotificationConfig):
        self.desktop_enabled = config.desktop_enabled
        self.whatsapp_enabled = config.whatsapp_enabled
        self.dnd_mode = config.dnd_mode
    
    async def send(self, message: str, priority: str = "normal"):
        if self.dnd_mode and priority != "critical":
            # Only WhatsApp in DND
            if self.whatsapp_enabled:
                await self._send_whatsapp(message)
        else:
            if self.desktop_enabled:
                await self._send_desktop(message)
            if self.whatsapp_enabled:
                await self._send_whatsapp(message)
```

2. Desktop notifications (using plyer or similar):
```python
def _send_desktop(self, message: str):
    from plyer import notification
    notification.notify(
        title="ROI",
        message=message,
        timeout=10
    )
```

3. WhatsApp integration (Business API or Twilio):
```python
async def _send_whatsapp(self, message: str):
    # Using WhatsApp Business API
    await httpx.post(
        f"https://graph.facebook.com/v19.0/{phone_id}/messages",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "messaging_product": "whatsapp",
            "to": self.user_phone,
            "type": "text",
            "text": {"body": message}
        }
    )
```

4. Settings integration:
```python
@router.patch("/settings/notifications")
async def update_notification_settings(settings: NotificationSettings):
    """Update notification preferences."""
```

**Success Criteria:**
- [ ] Desktop notifications working
- [ ] WhatsApp integration (if configured)
- [ ] DND mode respected
- [ ] Settings configurable

**Test:**
1. Trigger HITL event
2. Verify desktop notification appears
3. Enable DND mode
4. Verify only WhatsApp sent

---

### 5.7 Piper TTS Integration

**Goal:** Local text-to-speech for voice feedback

**Tasks:**
1. Install Piper:
```bash
uv add piper-tts
```

2. TTS service:
```python
class TTSService:
    def __init__(self, voice_model: str = "en_US-lessac-high"):
        import piper
        self.synthesizer = piper.Synthesizer(voice_model)
    
    def speak(self, text: str) -> bytes:
        """Convert text to audio."""
        return self.synthesizer.synthesize(text)
    
    async def speak_and_play(self, text: str):
        """Speak text through system audio."""
        audio = self.speak(text)
        # Play using sounddevice or similar
```

3. Integration points:
   - HITL notifications
   - Task completion announcements
   - Error alerts

4. Voice settings:
```python
@router.patch("/settings/voice")
async def update_voice_settings(settings: VoiceSettings):
    """Update voice preferences."""
```

**Success Criteria:**
- [ ] Piper TTS installed and working
- [ ] Can convert text to speech
- [ ] Audio plays on system
- [ ] Voice configurable

**Test:**
```python
tts = TTSService()
audio = tts.speak("Task completed successfully")
assert len(audio) > 0
```

---

### 5.8 Voice Input (STT Integration)

**Goal:** Local speech-to-text for voice conversations

**Tasks:**
1. Install faster-whisper:
```bash
uv add faster-whisper
```

2. STT service:
```python
from faster_whisper import WhisperModel

class STTService:
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        # model_size: tiny, base, small, medium, large
        self.model = WhisperModel(model_size, device=device)
        self.sample_rate = 16000

    async def transcribe(self, audio_data: bytes) -> str:
        """Convert audio to text."""
        segments, info = self.model.transcribe(
            audio_data,
            beam_size=5,
            language="en"
        )
        return " ".join([segment.text for segment in segments])

    def get_audio_duration(self, audio_data: bytes) -> float:
        """Get audio duration in seconds."""
        # For raw PCM audio
        return len(audio_data) / (self.sample_rate * 2)  # 16-bit
```

3. Audio capture service:
```python
import pyaudio

class AudioCapture:
    def __init__(self):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.audio = pyaudio.PyAudio()

    def start_recording(self, duration: int = 10):
        """Record audio for specified duration."""
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        frames = []
        for _ in range(int(self.rate / self.chunk * duration)):
            frames.append(stream.read(self.chunk))

        stream.stop_stream()
        stream.close()
        return b"".join(frames)

    def cleanup(self):
        self.audio.terminate()
```

4. Push-to-talk endpoint:
```python
@router.post("/voice/push-to-talk")
async def push_toTalk(audio: UploadFile):
    """Record and transcribe audio input."""
    # 1. Save uploaded audio
    audio_data = await audio.read()

    # 2. Transcribe
    text = await stt_service.transcribe(audio_data)

    # 3. Send to chat endpoint
    response = await process_message(text)

    # 4. Convert response to speech
    await tts_service.speak_and_play(response["content"])

    return {"transcript": text, "response": response}
```

5. Integration points:
   - Push-to-talk button in UI
   - Continuous streaming mode for long conversations
   - Audio format: WAV, 16kHz, 16-bit PCM

**Success Criteria:**
- [ ] faster-whisper installed and working
- [ ] Can transcribe audio accurately
- [ ] Push-to-talk endpoint functional
- [ ] Audio capture working
- [ ] English language support

**Test:**
```python
stt = STTService(model_size="base")
text = await stt.transcribe(audio_data)
assert len(text) > 0
print(f"Transcribed: {text}")
```

---

### 5.9 Wake Word Detection (Optional)

**Goal:** Hands-free voice activation with "Hey ROI"

**Tasks:**
1. Install porcupine:
```bash
uv add pvporcupine
```

2. Wake word service:
```python
import pvporcupine

class WakeWordDetector:
    def __init__(self, access_key: str):
        self.porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=["./wake_words/hey_roi.ppn"],  # Custom wake word
            sensitivities=[0.5]
        )
        self.sample_rate = self.porcupine.sample_rate
        self.frame_length = self.porcupine.frame_length

    def listen_for_wake_word(self, audio_frame: bytes) -> bool:
        """Check if wake word detected."""
        keyword_index = self.porcupine.process(audio_frame)
        return keyword_index >= 0

    def cleanup(self):
        self.porcupine.delete()
```

3. Always-listening service:
```python
class AlwaysListeningService:
    def __init__(self, wake_detector: WakeWordDetector):
        self.detector = wake_detector
        self.listening = False

    async def start_listening(self):
        """Continuously listen for wake word."""
        self.listening = True
        audio = AudioCapture()

        while self.listening:
            # Record short frame
            frame = audio.record_frame(duration=self.detector.frame_length)

            # Check for wake word
            if self.detector.listen_for_wake_word(frame):
                # Trigger voice input mode
                await self.on_wake_word_detected()

    async def on_wake_word_detected(self):
        """Handle wake word detection."""
        # 1. Play acknowledgment sound
        # 2. Start listening for command
        # 3. Transcribe and process
        # 4. Return to listening mode
```

4. Wake word settings:
```python
@router.patch("/settings/wake-word")
async def update_wake_word_settings(settings: WakeWordSettings):
    """Update wake word preferences."""
    # enable_wake_word: bool
    # sensitivity: float (0.0-1.0)
    # custom_wake_word: Optional[str]
```

5. Custom wake word training:
```python
# Use Picovoice Console to train custom "Hey ROI" wake word
# Download .ppn file and place in /wake_words/ directory
```

**Success Criteria:**
- [ ] Porcupine installed and working
- [ ] "Hey ROI" wake word detects accurately
- [ ] Always-listening service functional
- [ ] Wake word configurable in settings
- [ ] Low false positive rate

**Test:**
1. Say "Hey ROI"
2. Verify wake word detected
3. Verify voice input mode activates
4. Test sensitivity settings

---

### 5.10 Settings API

**Goal:** Centralized settings management

**Tasks:**
1. Settings schema:
```python
class Settings(BaseModel):
    # Models
    default_model: str
    available_models: list[ModelConfig]

    # Notifications
    desktop_notifications: bool
    whatsapp_enabled: bool
    whatsapp_phone: Optional[str]
    dnd_mode: bool

    # Voice Output (TTS)
    voice_enabled: bool
    voice_model: str
    voice_speed: float = 1.0

    # Voice Input (STT)
    stt_enabled: bool
    stt_model: str = "base"
    stt_language: str = "en"

    # Wake Word
    wake_word_enabled: bool = False
    wake_word_sensitivity: float = 0.5

    # Execution
    default_execution_env: str
    docker_timeout: int

    # Memory
    context_offload_threshold: float
```

2. Settings endpoints:
```python
@router.get("/settings")
async def get_settings():
    """Get current settings."""

@router.patch("/settings")
async def update_settings(settings: Settings):
    """Update settings."""

@router.post("/settings/reset")
async def reset_settings():
    """Reset to defaults."""

@router.patch("/settings/voice")
async def update_voice_settings(settings: VoiceSettings):
    """Update voice preferences."""

@router.patch("/settings/wake-word")
async def update_wake_word_settings(settings: WakeWordSettings):
    """Update wake word preferences."""
```

3. Settings storage:
   - PostgreSQL as primary storage
   - Local YAML file backup at `~/.roi/settings.yaml`
   - Runtime caching for performance

**Success Criteria:**
- [ ] All settings manageable via API
- [ ] Persisted to database
- [ ] Backup file created
- [ ] Voice settings included
- [ ] Wake word settings included

---

## Phase 5 Overall Success Criteria

- [ ] Mode system fully operational with auto-suggestions
- [ ] Context offloading at 40% working with notification
- [ ] Intent journal tracking preferences (hybrid auto+manual)
- [ ] Multi-model selection functional with smart suggestions
- [ ] Notifications (desktop + WhatsApp) working with DND
- [ ] Piper TTS speaking (real-time + pre-gen modes)
- [ ] **NEW:** Voice input (STT) working with faster-whisper
- [ ] **NEW:** Push-to-talk functional
- [ ] **NEW:** Wake word detection (optional) working
- [ ] Settings fully configurable including voice
- [ ] Conversational flow working (Alexa/Google Assistant style)

## Phase 5 Deliverable

**Demo:** Full system demo showing:
- Mode switching (auto-suggest + manual)
- Context offloading in action with notification
- Intent recall from journal
- Model switching with smart suggestions
- Voice conversation (STT + TTS)
- Push-to-talk interaction
- Optional wake word activation
- Notifications (desktop + WhatsApp with DND)
- Background tasks running while conversation continues
