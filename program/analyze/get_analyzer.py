from program.analyze.openai_analyzer import OpenAIAnalyzer
from program.analyze.o3_analyzer import O3Analyzer
from program.analyze.scaling_analyzer import ScalingAnalyzer

analyzers = {
    "4o": OpenAIAnalyzer(api_key="1234567", model="gpt-4o"),
}


def get_analyzer(model_name):
    return analyzers[model_name]
