from src.rag.pipeline import RAGPipeline
from ragprobe import RagEvaluator, DatasetGenerator
from ragprobe.adapters.base import EvalAdapter, RetrievalResult, GenerationResult
from src.llm_provider import generate
from src.config import get_config
from pathlib import Path
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename = "Logs/ragprobe_adapter.log",
    filemode = "a",
    encoding="utf-8")

logger = logging.getLogger(__name__)

def my_llm(prompt: str) -> str:
    return generate(prompt).content


class MyCustomAdapter(EvalAdapter):
    def __init__(self):
        self.rag = RAGPipeline()

    def retrieve(self, query: str) -> RetrievalResult:
        results = self.rag._retrieve(query)
        logger.info(f"Retrieved {len(results['documents'][0]) if results['documents'] else 0} documents for query: {query}")
        chunks = results["documents"][0] if results["documents"] else []
        scores = results["distances"][0] if results["distances"] else []
        return RetrievalResult(chunks=chunks, scores=scores)

    def generate(self, query: str, context: list[str]) -> GenerationResult:
        answer = self.rag._generate(query, context)
        return GenerationResult(answer=answer)


if __name__ == "__main__":
    # 1. Load chunks
    config = get_config()
    chunks_path = Path(config.extraction.output_dir).parent / "chunks" / "chunks.json"
    with open(chunks_path, "r") as f:
        logger.info(f"Loading chunks from {chunks_path}")
        chunks = json.load(f)
    
    logger.info(f"Successfully loaded {len(chunks)} chunks from {chunks_path}")

    # 2. Generate dataset from real chunks
    dataset = DatasetGenerator(llm_fn=my_llm).generate_from_chunks(chunks[:20], questions_per_chunk= 1, max_samples = 3)
    dataset.save("./eval/datasets/generated_dataset.json")
    logger.info(f"Generated {len(dataset)} samples")

    # 3. Evaluate
    adapter = MyCustomAdapter()
    evaluator = RagEvaluator(adapter=adapter)
    results = evaluator.evaluate(dataset)
    logger.info(f"Evaluation results: {results.summary()}")
