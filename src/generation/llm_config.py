"""
Thin wrapper around two ways of calling the LLM configured in configs/config.yaml
(llm.model / llm.provider):

  provider: "huggingface"  -> loads the model locally via transformers, 4-bit
                               quantized on GPU (falls back to float32 on CPU).
                               Needs a HF token with access to the model repo.
  provider: "groq"         -> calls Groq's free-tier hosted inference API
                               (no local compute needed, needs GROQ_API_KEY).

Both expose the same interface, at two granularities:

  generate_k_samples(prompt, k, ...) -> list[str]
      One prompt, k samples. Kept for single-prompt callers
      (threshold_calibration's LLM judge).

  generate_k_samples_batch(prompts, k, ...) -> list[list[str]]
      MANY prompts, k samples each - the batched path the generation
      stage uses. On GPU this is the important one: one
      model.generate() call decodes len(prompts) * k sequences
      together, so the GPU processes a real batch every step instead
      of sitting mostly idle on single-sequence decodes. Expected
      speedup over the per-claim loop is roughly the batch size
      (e.g. batch_size=10 claims x k=5 -> 50 sequences per step vs 5).
      Memory scales with batch * k * max_new_tokens, so lower
      batch_size if a bigger model OOMs.
"""
import os
import random
import time
import zlib
from dotenv import load_dotenv
from src.utils.logging_utils import get_logger

load_dotenv()  # ensures GROQ_API_KEY / HF_TOKEN are picked up from .env even
               # if this module is imported directly, not just via run_pipeline.py

logger = get_logger("llm_client")


class HuggingFaceLlamaClient:
    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.torch = torch
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN not set - gated model download will likely fail.")

        logger.info(f"Loading {model_name} via transformers (this can take a while)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                quantization_config=quant_config,
                # "auto" shards layers across ALL visible GPUs via accelerate
                # (required for 32b on 2xT4: ~17GB of 4-bit weights exceed one
                # 16GB card). On a single-GPU machine this collapses to cuda:0,
                # so it is a no-op there - never pin {"": 0}, which would OOM
                # large models by forcing every layer onto GPU 0.
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float32,
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            logger.warning("Running on CPU - generation will be slow for many samples.")
            self.model.to(self.device)

    def _generate(self, prompts: list, k: int, temperature: float,
                  top_p: float, max_new_tokens: int) -> list:
        """Shared implementation for both the single-prompt and batched
        entry points. `prompts` is a list of user-message strings; returns
        a list of per-prompt sample lists (len(prompts) entries, k each).

        Batching strategy: apply the chat template to each prompt
        individually (templates are model-specific and not batchable),
        left-pad tokenized into ONE tensor of shape
        (len(prompts), max_prompt_len) - left padding is required so
        every sequence ends at the same position and generation
        continues correctly for all rows - then pass
        num_return_sequences=k to generate() so the whole batch of
        len(prompts) * k sequences decodes in a single call, GPU-wide.

        Left-padding note for correctness: sampling *is* affected by
        pad-position-dependent numerical drift, so left padding makes
        batched results not bit-identical to unbatched runs - a
        distribution-equivalent, not deterministic, equivalence. That's
        fine for this pipeline: we're sampling at temperature 0.7-0.9
        anyway, so any per-row difference is far below run-to-run
        sampling noise. (For reference, vLLM etc. add batches without
        padding at all via PagedAttention for exactly this reason -
        this is the same tradeoff, just with plain padding.)
        """
        if not prompts:
            return []

        chat_lists = [[{"role": "user", "content": p}] for p in prompts]
        texts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in chat_lists
        ]
        # tokenizer.pad_token must be a real padding token for left padding
        # to work (generation later uses eos as pad_token_id, which is fine
        # there, but tokenizer() itself needs a valid pad_token set).
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False,
        ).to(self.device)

        # temperature=0 means "deterministic/greedy" (used by the threshold-
        # calibration LLM-judge for consistent labeling). transformers'
        # generate() rejects temperature=0 under sampling mode (it divides
        # logits by temperature, so 0 would be a divide-by-zero), so route
        # that case to do_sample=False instead of passing temperature=0 in.
        do_sample = temperature > 0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_return_sequences=k,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        elif k > 1:
            # Greedy decoding is deterministic, so num_return_sequences>1
            # would just produce k identical copies - not useful, and some
            # transformers versions warn/error on this combination.
            logger.warning(
                f"temperature={temperature} (greedy) with k={k}>1 requested - "
                f"greedy decoding is deterministic, so only 1 unique sample will "
                f"be generated and repeated {k} times."
            )

        # Local decode rarely fails, but a sporadic CUDA/driver hiccup on a
        # multi-hour run shouldn't kill the whole generation stage. Two
        # attempts max: OOM is re-raised immediately (retrying an OOM just
        # OOMs again - lower generation.batch_size instead).
        outputs = None
        for attempt in range(2):
            try:
                outputs = self.model.generate(input_ids=encoded["input_ids"],
                                              attention_mask=encoded["attention_mask"],
                                              **gen_kwargs)
                break
            except Exception as e:
                if "out of memory" in str(e).lower():
                    raise
                if attempt == 0:
                    logger.warning(f"model.generate() failed ({e}) - retrying once...")
                    if self.torch.cuda.is_available():
                        self.torch.cuda.empty_cache()
                else:
                    raise
        # outputs shape: (len(prompts) * k, full_seq_len). generate() with
        # num_return_sequences=k expands each input row into k adjacent
        # consecutive output rows, so outputs[i*k : (i+1)*k] are the k
        # samples for prompts[i].
        prompt_len = encoded["input_ids"].shape[1]
        all_samples = [
            self.tokenizer.decode(out[prompt_len:], skip_special_tokens=True).strip()
            for out in outputs
        ]
        return [all_samples[i * k:(i + 1) * k] for i in range(len(prompts))]

    def generate_k_samples(self, prompt: str, k: int, temperature: float,
                            top_p: float, max_new_tokens: int) -> list[str]:
        return self._generate([prompt], k, temperature, top_p, max_new_tokens)[0]

    def generate_k_samples_batch(self, prompts: list, k: int, temperature: float,
                                 top_p: float, max_new_tokens: int) -> list:
        return self._generate(prompts, k, temperature, top_p, max_new_tokens)


class GroqLlamaClient:
    def __init__(self, model_name: str, reasoning_effort: str = None, seed: int = None,
                 max_retries: int = 3, retry_backoff: float = 1.0):
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set. Get a free key at console.groq.com")
        self.client = Groq(api_key=api_key)
        # Uses whatever model name is set in configs/config.yaml under llm.model.
        # Must match a model name available in your Groq account -
        # check with: client.models.list()
        self.model_name = model_name
        # Optional Groq-side reasoning control (e.g. "none" to fully disable
        # thinking on hybrid models like qwen3.6-27b, or "low" to minimize it
        # on reasoning-only models like gpt-oss). None (default) preserves
        # today's behavior: the parameter is omitted from the API call.
        # Set via the model entry in configs/config.yaml (see the judge entry).
        self.reasoning_effort = reasoning_effort
        if reasoning_effort is not None:
            logger.info(f"Groq reasoning_effort={reasoning_effort!r} for {model_name}")
        # Optional API-side seed for reproducible sampling (Groq honors it;
        # None omits it entirely). Mirrors llm.seed, which seeds local RNGs
        # for HF. IMPORTANT: the base seed is NOT forwarded verbatim per
        # call - see _derived_seed (forwarding the same seed to all k calls
        # of the same prompt would make a seed-honoring API return k
        # IDENTICAL completions, collapsing sample diversity to ~0).
        self.seed = seed
        # Retry policy for transient API failures (429 rate limits, timeouts,
        # 5xx). max_retries is TOTAL attempts; retry_backoff scales the
        # exponential delay between attempts (backoff * 2^attempt + jitter).
        self.max_retries = max(1, max_retries)
        self.retry_backoff = retry_backoff

    def _derived_seed(self, prompt: str, sample_index: int):
        """Per-call seed derived from the configured base seed, the prompt,
        and the sample index: (seed + crc32(prompt) + sample_index) % (2^31 - 1).

        Why not just forward the base seed on every call: k samples of the
        same prompt share the prompt AND all sampling parameters - identical
        seeds too would make every completion a duplicate of the first.

        Why a prompt hash instead of a running call counter: the derived seed
        stays stable per (prompt, sample_index) across resumed runs and
        batch-size changes, so identical reruns reproduce exactly while
        different samples still differ. A counter shifts every seed after a
        resume. crc32 is stdlib, stable across processes, and int32-safe
        for the API. Returns None (seed omitted from the request) when no
        base seed is configured."""
        if self.seed is None:
            return None
        return (self.seed + zlib.crc32(prompt.encode("utf-8")) + sample_index) % (2**31 - 1)

    def _create_kwargs(self, prompt: str, temperature: float,
                       top_p: float, max_new_tokens: int, sample_index: int = 0) -> dict:
        """Shared chat.completions.create kwargs; reasoning_effort is only
        included when configured, and the seed is the per-call DERIVED seed
        (see _derived_seed) - omitted entirely when no base seed is set, so
        default calls are byte-identical to before."""
        kwargs = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        seed = self._derived_seed(prompt, sample_index)
        if seed is not None:
            kwargs["seed"] = seed
        return kwargs

    def _create_with_retry(self, **kwargs):
        """One chat.completions.create call with retries for transient API
        failures (429 rate limits on the free tier, timeouts, 5xx). Without
        this, a single hiccup kills the whole batch (and generation run).

        Retries max_retries TOTAL attempts with exponential backoff
        (retry_backoff * 2^attempt + up-to-1s jitter), then re-raises the
        last error. Auth/config errors (401/404) will also be retried and
        fail the same way - they surface after the retries, unchanged."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_backoff * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Groq call failed (attempt {attempt + 1}/{self.max_retries}): "
                        f"{e} - retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
        raise last_error

    def generate_k_samples(self, prompt: str, k: int, temperature: float,
                            top_p: float, max_new_tokens: int) -> list[str]:
        samples = []
        for i in range(k):
            # Groq's chat API returns one completion per call; loop for k samples.
            # sample_index=i gives each call a distinct derived seed (see
            # _derived_seed) so k calls don't return k identical completions.
            response = self._create_with_retry(
                **self._create_kwargs(prompt, temperature, top_p, max_new_tokens, sample_index=i),
            )
            samples.append(response.choices[0].message.content.strip())
        return samples

    def generate_k_samples_batch(self, prompts: list, k: int, temperature: float,
                                 top_p: float, max_new_tokens: int) -> list:
        """Groq's chat API takes one prompt per call. Batching here means
        one call per (prompt, sample) pair via a thread pool, which still
        parallelizes network latency instead of paying it sequentially for
        every sample. Same return contract as the HF client: one list of k
        sample strings per prompt, in input order. Each call carries its own
        derived seed (base seed + prompt hash + within-prompt sample index),
        so the k samples of a prompt differ from each other."""
        from concurrent.futures import ThreadPoolExecutor

        jobs = [(p, i, j) for i, p in enumerate(prompts) for j in range(k)]

        def _call(job):
            prompt, _, sample_index = job
            response = self._create_with_retry(
                **self._create_kwargs(prompt, temperature, top_p, max_new_tokens,
                                      sample_index=sample_index),
            )
            return response.choices[0].message.content.strip()

        # Cap the pool: one thread per (prompt, sample) job is fine for the
        # small per-batch fan-out, but an uncapped pool explodes if batch
        # sizes grow (a full 907-claim run would otherwise spawn thousands).
        with ThreadPoolExecutor(max_workers=min(32, max(1, len(jobs)))) as pool:
            flat = list(pool.map(_call, jobs))

        return [flat[i * k:(i + 1) * k] for i in range(len(prompts))]


def get_llm_client(config: dict):
    provider = config["llm"]["provider"]
    model_name = config["llm"]["model"]
    if provider == "huggingface":
        return HuggingFaceLlamaClient(model_name)
    elif provider == "groq":
        # reasoning_effort is Groq-API-specific (local HF generate() has no
        # such parameter); forwarded only when a model entry sets it, e.g.
        # the calibration judge entry disabling thinking on qwen3.6-27b.
        # seed mirrors llm.seed for reproducible API-side sampling.
        # max_retries/retry_backoff tune the transient-failure policy
        # (llm.max_retries / llm.retry_backoff, defaulting to 3 / 1.0s).
        return GroqLlamaClient(model_name,
                               reasoning_effort=config["llm"].get("reasoning_effort"),
                               seed=config["llm"].get("seed"),
                               max_retries=config["llm"].get("max_retries", 3),
                               retry_backoff=config["llm"].get("retry_backoff", 1.0))
    else:
        raise ValueError(f"Unknown provider: {provider}")