import os
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sec_api import QueryApi, ExtractorApi
from dotenv import load_dotenv

load_dotenv()


class FinancialReportSummarizer:
    """
    A class to handle 10-K financial report summarization using T5 model.
    Extracts the Management Discussion & Analysis (MD&A) section from SEC filings
    and generates a concise summary.
    """

    def __init__(self):
        """
        Initializes the T5 model, tokenizer, and SEC API client.
        """
        print("Loading T5 model and tokenizer...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        model_name = "t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        print("T5 Model loaded successfully.")

        # Load SEC API key from environment
        self.sec_api_key = os.environ.get("SEC_API_KEY")
        if not self.sec_api_key:
            raise RuntimeError("SEC_API_KEY not found in .env file.")

    def get_10k_summary(self, ticker: str) -> str:
        """
        Fetches the latest 10-K filing for a ticker and returns a T5-generated summary
        of the MD&A section.

        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')

        Returns:
            str: Summary of the 10-K filing's MD&A section
        """
        try:
            print(f"Fetching 10-K filing for {ticker}...")

            # Query SEC API for the latest 10-K
            queryApi = QueryApi(api_key=self.sec_api_key)
            query = {
                "query": {
                    "query_string": {
                        "query": f'ticker:{ticker} AND formType:"10-K"'
                    }
                },
                "from": "0",
                "size": "1",
                "sort": [{"filedAt": {"order": "desc"}}],
            }
            filings = queryApi.get_filings(query)

            if not filings.get("filings") or len(filings["filings"]) == 0:
                return f"No recent 10-K filing found for {ticker}."

            filing_url = filings["filings"][0]["linkToFilingDetails"]
            print(f"Found filing: {filing_url}")

            # Extract MD&A section (Section 7)
            extractorApi = ExtractorApi(self.sec_api_key)
            mda_text = extractorApi.get_section(
                filing_url=filing_url, section="7", return_type="text"
            )

            # Generate summary using T5
            print(f"Generating summary for {ticker}...")
            summary_input_text = "summarize: " + mda_text[:4000]
            inputs = self.tokenizer.encode(
                summary_input_text,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
            ).to(self.device)

            summary_ids = self.model.generate(
                inputs,
                max_length=2500,
                min_length=1000,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            print(f"✅ Summary generated for {ticker}")
            return summary

        except Exception as e:
            error_msg = f"Could not process {ticker}. Reason: {str(e)}"
            print(error_msg)
            return error_msg
