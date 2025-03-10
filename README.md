# Resume ATS Analyzer

This project provides a Python-based tool to analyze resumes against job descriptions, providing an ATS (Applicant Tracking System) score and feedback to help job seekers improve their resumes. It leverages NLP techniques, including SpaCy, NLTK, and optionally Google's Gemini API, to provide comprehensive analysis and suggestions.

## Features

*   **ATS Score:** Calculates an overall ATS compatibility score based on keyword matching, skills, experience, education, and readability.
*   **Section-Specific Feedback:** Provides detailed feedback and suggestions for improvement for the following resume sections:
    *   Skills
    *   Experience
    *   Education
    *   Projects
    *   Other (uncategorized content)
*   **Keyword Analysis:** Identifies missing and present keywords from the job description in the resume.
*   **Readability Analysis:** Assesses the resume's readability using metrics like Flesch Reading Ease and Flesch-Kincaid Grade Level, checking for passive voice, complex words, and sentence structure.
*   **Industry-Specific Keyword Matching:** Includes a database of industry-specific keywords (data science, software development, marketing, finance, healthcare) to improve matching accuracy.
*   **AI-Powered Suggestions (Optional):** Integrates with Google's Gemini API (requires a Google API key) to provide AI-generated suggestions for improving each section.
*   **JSON Output:** Produces a structured JSON output of the analysis, making it easy to integrate with other applications or display the results.
*   **Command-Line Interface:** Easy-to-use command-line interface for analyzing resumes.
* **Handles multiple resume format**: .pdf, .png, .jpg, .txt

## Requirements

*   Python 3.7+
*   Tesseract OCR (for PDF and image processing):
    *   **Windows:** Download from [here](https://github.com/UB-Mannheim/tesseract/wiki) and add to your PATH.
    *   **macOS:** `brew install tesseract`
    *   **Linux (Debian/Ubuntu):** `sudo apt-get install tesseract-ocr`
*   Required Python packages (install using `pip`):

    ```
    pip install -r requirements.txt
    ```

    The `requirements.txt` file should contain:

    ```
    pdf2image
    Pillow
    pytesseract
    nltk
    spacy
    scikit-learn
    python-textstat
    google-generativeai
    python-dotenv
    ```

*   NLTK Data (downloaded automatically on first run, but you can also do it manually):

    ```bash
    python -m nltk.downloader punkt stopwords wordnet
    ```

*   SpaCy Model (downloaded automatically, or manually):

    ```bash
    python -m spacy download en_core_web_sm
    ```

*   Google API Key (Optional):
    *   Obtain a key from [Google Cloud Console](https://makersuite.google.com/app/apikey).
    *   Set the `GOOGLE_API_KEY` environment variable.  The recommended way is to create a `.env` file in the project root and add:

        ```
        GOOGLE_API_KEY=YOUR_API_KEY
        ```

        Then, install `python-dotenv` (`pip install python-dotenv`) and add these lines at the top of `main.py`:

        ```python
        from dotenv import load_dotenv
        import os

        load_dotenv()
        ```

        **Do not commit your `.env` file to version control.** Add `.env` to your `.gitignore` file.
    * Alternatively, use a VS Code `launch.json` configuration to set `GOOGLE_API_KEY`.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/NamanSingh69/Resume-Analyzer
    cd Resume-Analyzer
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On macOS/Linux
    .venv\Scripts\activate    # On Windows
    ```

3.  **Install Tesseract OCR** (see Requirements section above).

4.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up the Google API key (optional):**  Follow the instructions in the "Requirements" section.

## Usage

```bash
python main.py <path_to_resume> <path_to_job_description>
```

*   `<path_to_resume>`:  Path to the resume file (PDF, JPG, PNG, or TXT).
*   `<path_to_job_description>`: Path to a plain text file (`.txt`) containing the job description.

**Example:**

```bash
python main.py resumes/my_resume.pdf job_descriptions/data_scientist.txt
```

The script will output a JSON object containing the analysis results to the console.

## Example Output (JSON)

```json
{
  "ats_score": 72.5,
  "overall_feedback": "Good resume, but consider making some improvements. Resume matches 8 of 15 key terms.",
  "section_feedback": {
    "skills": {
      "score": 85.0,
      "feedback": "Matched 7 of 15 job keywords and 10 of 25 industry keywords.",
      "suggestions": [
        "Consider adding key skills like: machine learning, deep learning, data visualization"
      ]
    },
    "experience": {
      "score": 68.3,
      "feedback": "Your 4 years of experience is less than the required 5+ years. Your experience is not closely aligned with the job responsibilities.",
      "suggestions": [
        "Highlight other relevant experience or projects to compensate for the experience gap.",
        "Quantify your accomplishments with specific metrics (%, $, numbers).",
        "Tailor your experience bullet points to match the job responsibilities."
      ]
    },
    "education": {
      "score": 90.0,
      "feedback": "Your Master's degree meets the required education level.",
      "suggestions": [
        "Add relevant coursework, projects, or academic achievements that relate to the job."
      ]
    },
      "projects": {
      "score": 60.2,
      "feedback": "Your project descriptions lack structured formatting.",
      "suggestions": [
        "Use bullet points to clearly describe each project's purpose, your role, and technologies used."
      ]
    },
    "other": {
      "score": 50.0,
      "feedback": "This section contains information not categorized elsewhere. Ensure all relevant information is placed in standard sections.",
      "suggestions": [
        "Consider moving relevant details to standard sections (Skills, Experience, Education, Projects)."
      ]
    }
  },
  "missing_sections": [],
  "keywords_feedback": {
    "missing_keywords": [
      "machine learning",
      "deep learning",
      "data visualization",
      "tensorflow",
      "pytorch"
    ],
    "present_keywords": [
      "python",
      "sql",
      "data analysis",
      "statistics",
      "r"
    ]
  }
}
```

## Contributing

Contributions are welcome!  Please feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  (You'll need to create a `LICENSE` file and put the MIT License text in it.)

## Disclaimer

This tool provides an *estimate* of ATS compatibility.  It is not a guarantee of success in the job application process.  ATS systems vary widely, and human review is always a factor.  Use this tool as a guide to improve your resume, but always tailor your resume to each specific job application.
