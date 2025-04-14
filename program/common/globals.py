class Args:
    ENABLE_LOGGING = True
    LOG_TO_FILE = True
    LOG_FILENAME = "app.log"
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    CONTEXT_FOLDER = "../CIVDataset"
    META_DATA_PATH = "../data/CIVDataset-metadata.csv"
    HISTORY_FOLDER = "../load_history"

    DEEPSEEK_R_COLOR = "#5BB5AC"
    DEEPSEEK_NR_COLOR = "#8bdad2"
    QWEN_R_COLOR = "#D8B365"
    QWEN_NR_COLOR = "#ebcb88"
    LLAMA_R_COLOR = "#DE526C"
    LLAMA_NR_COLOR = "#f37e94"
    GPT_R_COLOR = "#82B0D2"
    GPT_NR_COLOR = "#9bbede"
