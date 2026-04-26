"""
resume_tailor.py — Tailors your resume to a specific job description using Claude AI.

HOW TO USE:
  1. Add your ANTHROPIC_API_KEY to a .env file in this directory:
       ANTHROPIC_API_KEY=sk-ant-...
  2. Put your resume text in base_resume.txt (same directory)
  3. Run:  python resume_tailor.py
  4. Paste a job listing URL when prompted — the script fetches and parses it automatically
"""

import os
import anthropic
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


# -----------------------------------------------------------------------------
# Load environment variables (.env file must contain ANTHROPIC_API_KEY)
# -----------------------------------------------------------------------------
load_dotenv()


# -----------------------------------------------------------------------------
# URL fetching: prompt the user for a job listing URL, download the page, and
# extract all visible text by stripping HTML tags with BeautifulSoup.
# -----------------------------------------------------------------------------
def fetch_job_description(url: str) -> str:
    """
    Fetch a job listing page and return its visible text content.

    Strips all HTML tags using BeautifulSoup so Claude receives clean,
    readable text rather than raw markup.  A browser-like User-Agent header
    is sent because some job boards block the default requests agent.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    # Attempt to download the page; raises on network errors or bad status codes
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # 4xx / 5xx → HTTPError
    except requests.exceptions.MissingSchema:
        raise ValueError(
            f"'{url}' is not a valid URL. "
            "Make sure it starts with http:// or https://"
        )
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Could not reach '{url}'. "
            "Check the URL and your internet connection."
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Request to '{url}' timed out after 15 seconds.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Page returned an error: {e}")

    # Parse the HTML and extract all visible text, collapsing excess whitespace
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script and style blocks — they contain no readable job content
    for tag in soup(["script", "style"]):
        tag.decompose()

    # get_text uses newlines as separators; strip() removes leading/trailing space
    text = soup.get_text(separator="\n").strip()

    # Collapse runs of blank lines so the text is compact but still readable
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)

    return cleaned


def read_resume(filepath: str = "base_resume.txt") -> str:
    """Read the base resume from a text file in the current directory."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"'{filepath}' not found. "
            "Create base_resume.txt in the same directory as this script."
        )
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"'{filepath}' is empty. Please add your resume text.")
    return content


def tailor_resume(resume_text: str, job_description: str) -> str:
    """
    Send the resume and job description to Claude and return a tailored resume.

    Prompt caching is used on both the system prompt and the resume text.
    These rarely change between runs, so repeated calls with different job
    descriptions reuse the cached tokens — saving time and API costs.
    """
    # The Anthropic client automatically reads ANTHROPIC_API_KEY from the environment
    client = anthropic.Anthropic()

    # System prompt: tells Claude how to behave for this task
    system_prompt = (
        "You are an expert resume writer and career coach. "
        "Your task is to tailor resumes to specific job descriptions.\n\n"
        "Rules you must follow:\n"
        "- NEVER fabricate or exaggerate experience, skills, or achievements\n"
        "- NEVER add qualifications or certifications the person does not have\n"
        "- Only reorder, reframe, or emphasize information that already exists\n"
        "- Naturally incorporate relevant keywords from the job description\n"
        "- Adjust the summary or objective section to match the role\n"
        "- Highlight the most relevant experience for this specific job\n"
        "- Preserve the original resume's overall structure and formatting\n"
        "- Return ONLY the complete tailored resume — no explanations or commentary"
    )

    # Build the API request
    # cache_control marks content that should be cached between API calls.
    # The system prompt and resume are stable, so we cache them.
    # The job description changes per application, so it is NOT cached.
    response = client.messages.create(
        model="claude-sonnet-4-20250514",  # explicitly requested model
        max_tokens=4096,

        # Cache the system prompt — it never changes between runs
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],

        messages=[
            {
                "role": "user",
                "content": [
                    # Cache the resume text — same resume, many different job descriptions
                    {
                        "type": "text",
                        "text": f"Here is my base resume:\n\n{resume_text}",
                        "cache_control": {"type": "ephemeral"},
                    },
                    # No cache on the job description — this changes every run
                    {
                        "type": "text",
                        "text": (
                            f"Here is the job description I am applying to:\n\n"
                            f"{job_description}\n\n"
                            "Please tailor my resume for this position. "
                            "Keep all information completely truthful and accurate. "
                            "Return only the finished, tailored resume."
                        ),
                    },
                ],
            }
        ],
    )

    # The response content is a list of blocks; the first block holds the text
    return response.content[0].text


def main():
    # ── Step 1: Load the resume ───────────────────────────────────────────────
    try:
        resume = read_resume()
        print(f"✓ Resume loaded ({len(resume):,} characters)")
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
        return

    # ── Step 2: Fetch the job description from a URL ─────────────────────────
    # Prompt interactively so the user can paste any job listing URL at runtime
    print("\nPaste the job listing URL and press Enter:")
    url = input("  URL: ").strip()

    if not url:
        print("\nError: No URL entered. Please run the script again and provide a URL.")
        return

    print(f"\nFetching job listing from: {url}")
    try:
        job_desc = fetch_job_description(url)
    except (ValueError, ConnectionError, TimeoutError, RuntimeError) as e:
        print(f"\nError: {e}")
        return

    if not job_desc:
        print(
            "\nError: The page was fetched successfully but contained no readable text. "
            "Try a different URL or paste the job description directly into the script."
        )
        return

    # Confirm how much text was extracted so the user can judge quality
    print(f"✓ Job description fetched ({len(job_desc):,} characters)")

    # ── Step 3: Call the Claude API ───────────────────────────────────────────
    print("\nSending to Claude for tailoring (this may take a moment)...\n")
    try:
        tailored = tailor_resume(resume, job_desc)

    except anthropic.AuthenticationError:
        print("Error: Invalid or missing API key.")
        print("Make sure ANTHROPIC_API_KEY is set in your .env file.")
        return

    except anthropic.RateLimitError:
        print("Error: API rate limit reached. Please wait a moment and try again.")
        return

    except anthropic.APIConnectionError:
        print("Error: Could not connect to the Anthropic API.")
        print("Check your internet connection and try again.")
        return

    except anthropic.APIStatusError as e:
        print(f"API error (HTTP {e.status_code}): {e.message}")
        return

    # ── Step 4: Print the result ──────────────────────────────────────────────
    separator = "=" * 64
    print(separator)
    print("TAILORED RESUME")
    print(separator)
    print(tailored)
    print(separator)


if __name__ == "__main__":
    main()
