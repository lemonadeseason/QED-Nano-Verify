#!/usr/bin/env python3
"""
Quick test: Call Azure GPT-5.2 with the same grading prompt used in training,
verify that score parsing works.

Usage:
  export AZURE_GRADER_ENDPOINT="https://msncompanioneu2.cognitiveservices.azure.com/"
  export AZURE_GRADER_KEY="<key>"
  export AZURE_GRADER_DEPLOYMENT="gpt-5-2"
  conda run --no-capture-output -n pipeline-rl python test_azure_grader.py
"""
import os
import re
import sys
import time
import asyncio

# Ensure the training code is importable
sys.path.insert(0, os.path.dirname(__file__))

from openai import AzureOpenAI


def get_client():
    endpoint = os.getenv("AZURE_GRADER_ENDPOINT")
    api_key = os.getenv("AZURE_GRADER_KEY")
    api_version = os.getenv("AZURE_GRADER_API_VERSION", "2024-12-01-preview")
    if not endpoint or not api_key:
        print("ERROR: Set AZURE_GRADER_ENDPOINT and AZURE_GRADER_KEY")
        sys.exit(1)
    return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)


def test_raw_api():
    """Test 1: Raw API call to see response structure."""
    print("=" * 60)
    print("TEST 1: Raw Azure GPT-5.2 API response structure")
    print("=" * 60)

    client = get_client()
    model = os.getenv("AZURE_GRADER_DEPLOYMENT", "gpt-5-2")

    # Use the same prompt format as the v1 evaluator
    prompt = """You are an **expert math proof grader**.

### Task
Grade the following proof on a scale of 0-7.

**Problem**: Prove that for any positive integer n, the sum 1+2+...+n = n(n+1)/2.

**Marking Scheme**:
# Proof by induction (7 points)
Description: Complete proof by mathematical induction with base case, inductive hypothesis, and inductive step.

**Proof Solution**:
We prove by induction. Base case: n=1, sum=1=1*2/2. Inductive step: assume true for k, then 1+2+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k+2)/2. QED.

### Output Format
Respond with **only** well-formed XML:
<score>0</score>
<assessment>Brief assessment</assessment>
<errors>List of errors</errors>
"""

    for attempt in range(5):
        try:
            print(f"\n  Attempt {attempt + 1}...")
            t0 = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4096,
            )
            elapsed = time.time() - t0
            print(f"  API call succeeded in {elapsed:.1f}s")

            choice = response.choices[0]
            msg = choice.message

            # Check all content fields
            content = msg.content
            reasoning_content = getattr(msg, "reasoning_content", "ATTR_NOT_FOUND")
            reasoning = getattr(msg, "reasoning", "ATTR_NOT_FOUND")

            print(f"\n  message.content type:            {type(content)}")
            print(f"  message.content is None:         {content is None}")
            if content:
                print(f"  message.content length:          {len(content)}")
                print(f"  message.content (first 500):")
                print(f"    {content[:500]}")
            else:
                print(f"  message.content value:           {content!r}")

            print(f"\n  message.reasoning_content type:  {type(reasoning_content)}")
            if isinstance(reasoning_content, str) and reasoning_content != "ATTR_NOT_FOUND":
                print(f"  message.reasoning_content len:   {len(reasoning_content)}")
                print(f"  message.reasoning_content (first 300):")
                print(f"    {reasoning_content[:300]}")
            else:
                print(f"  message.reasoning_content:       {reasoning_content!r}")

            print(f"\n  message.reasoning type:          {type(reasoning)}")
            if reasoning != "ATTR_NOT_FOUND" and reasoning is not None:
                print(f"  message.reasoning:               {str(reasoning)[:300]}")

            # Test score parsing
            output_text = content or ""
            if not output_text:
                rc = getattr(msg, "reasoning_content", None) or ""
                if rc:
                    output_text = rc
                    print(f"\n  >> Using reasoning_content as fallback!")

            match = re.search(r"<score>\s*(\d+)\s*</score>", output_text) or re.search(r"<points>\s*(\d+)", output_text)
            if match:
                score = int(match.group(1))
                print(f"\n  >> SCORE PARSED SUCCESSFULLY: {score}/7")
            else:
                print(f"\n  >> SCORE PARSING FAILED! No <score> or <points> tag found")
                print(f"  >> output_text length: {len(output_text)}")
                if output_text:
                    print(f"  >> output_text (full):\n{output_text}")

            # Usage info
            if response.usage:
                print(f"\n  Usage: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")

            return True

        except Exception as e:
            print(f"  Failed: {e}")
            if attempt < 4:
                wait = 10
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)

    print("  All attempts failed!")
    return False


def test_verify_proof():
    """Test 2: Call verify_proof() directly with Azure grader."""
    print("\n" + "=" * 60)
    print("TEST 2: verify_proof() integration test")
    print("=" * 60)

    # Set up environment
    os.environ.setdefault("AZURE_GRADER_ENDPOINT", "https://msncompanioneu2.cognitiveservices.azure.com/")
    os.environ.setdefault("AZURE_GRADER_DEPLOYMENT", "gpt-5-2")

    try:
        from pipelinerl.domains.math.verifier_api import verify_proof, parse_schema
    except ImportError as e:
        print(f"  Cannot import verify_proof: {e}")
        print("  Make sure you're in the training directory and pipeline-rl env is active")
        return False

    problem = "Prove that for any positive integer n, the sum 1+2+...+n = n(n+1)/2."
    ref_solution = "Proof by induction. Base: n=1. Step: assume for k, prove for k+1."
    schema = "# Proof by induction (7 points)\nDescription: Complete proof by mathematical induction."
    generation = (
        "We prove by induction on n.\n"
        "Base case: n=1. LHS=1, RHS=1*2/2=1. True.\n"
        "Inductive step: Assume 1+2+...+k = k(k+1)/2 for some k>=1.\n"
        "Then 1+2+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2+1) = (k+1)(k+2)/2.\n"
        "This completes the induction. QED."
    )
    model = os.getenv("AZURE_GRADER_DEPLOYMENT", "gpt-5-2")

    for attempt in range(3):
        try:
            print(f"\n  Attempt {attempt + 1}: calling verify_proof()...")
            t0 = time.time()
            result = asyncio.run(
                verify_proof(
                    problem=problem,
                    ref_solution=ref_solution,
                    schema=schema,
                    generation=generation,
                    prompt_name="v1",
                    model=model,
                    sampling_kwargs={"max_output_tokens": 4096},
                    log_wandb_metrics=False,
                )
            )
            elapsed = time.time() - t0
            print(f"  verify_proof completed in {elapsed:.1f}s")
            print(f"  Score: {result.score}/7")
            print(f"  Metrics: {result.metrics}")
            if result.score > 0:
                print(f"\n  >> SUCCESS: verify_proof correctly parsed score={result.score}")
                return True
            else:
                print(f"\n  >> WARNING: score=0 — check logs above for diagnostics")
                return False
        except Exception as e:
            print(f"  Failed: {e}")
            if attempt < 2:
                time.sleep(10)

    print("  All attempts failed!")
    return False


if __name__ == "__main__":
    print("Azure GPT-5.2 Grader Test")
    print(f"Endpoint: {os.getenv('AZURE_GRADER_ENDPOINT', 'NOT SET')}")
    print(f"Deployment: {os.getenv('AZURE_GRADER_DEPLOYMENT', 'NOT SET')}")
    print()

    ok1 = test_raw_api()
    ok2 = test_verify_proof() if ok1 else False

    print("\n" + "=" * 60)
    print(f"RESULTS: raw_api={'PASS' if ok1 else 'FAIL'}, verify_proof={'PASS' if ok2 else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if (ok1 and ok2) else 1)
