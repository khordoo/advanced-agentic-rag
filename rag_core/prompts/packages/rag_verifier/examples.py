from pydantic import BaseModel, Field

from rag_core.prompts.packages.rag_verifier.schema import RagVerifierResponse


class RagVerifierExample(BaseModel):
    documents: str = Field(description="The documents")
    answer: str = Field(description="The generated answer to verify")
    response: RagVerifierResponse = Field(description="The expected verifier response")


EXAMPLES = [
    RagVerifierExample(
        documents="""
[Asset: report_a1b2c3d4.pdf] [Section: Data Table] [Location: Page 5]

| ($ millions) | 2024  | 2023  | Change vs. 2023 |
|--------------|-------|-------|-----------------|
| Net earnings | 2,963 | 3,178 | (7)%           |
""",
        answer="The year-over-year change in net earnings was -7%, and net earnings were $2,963 million.",
        response=RagVerifierResponse(
            grounded=True,
            reason="The table shows net earnings of 2,963 under a ($ millions) header and a change of (7)%, which supports $2,963 million and -7%.",
        ),
    ),
    RagVerifierExample(
        documents="""
[Asset: report_a1b2c3d4.pdf] [Section: Data Table] [Location: Page 5]

| ($ millions) | 2024  | 2023  | Change vs. 2023 |
|--------------|-------|-------|-----------------|
| Net earnings | 2,963 | 3,178 | (7)%           |
""",
        answer="The year-over-year change in net earnings was 7%, and net earnings were $2,963 million.",
        response=RagVerifierResponse(
            grounded=False,
            reason="The table shows a change of (7)%, which indicates negative 7%, not positive 7%.",
        ),
    ),
    RagVerifierExample(
        documents="""
[Asset: report_a1b2c3d4.pdf] [Section: Data Table] [Location: Page 5]

| Segment | Price | Foreign Exchange | Mix | Net Sales Growth |
|---------|-------|------------------|-----|------------------|
| Beauty  | 4%    | (2)%             | (1)%| 1%               |
""",
        answer="Beauty net sales growth was driven by a 4% pricing benefit, offset by unfavorable foreign exchange of 2% and unfavorable mix of 1%.",
        response=RagVerifierResponse(
            grounded=True,
            reason="The answer matches the Beauty row values and correctly interprets parentheses as unfavorable negative impacts.",
        ),
    ),
    RagVerifierExample(
        documents="""
[Asset: report_a1b2c3d4.pdf] [Section: Data Table] [Location: Page 5]

| Segment | Price | Foreign Exchange | Mix | Net Sales Growth |
|---------|-------|------------------|-----|------------------|
| Beauty  | 4%    | (2)%             | (1)%| 1%               |
""",
        answer="Beauty net sales growth was driven by a 4% pricing benefit and favorable foreign exchange of 2%.",
        response=RagVerifierResponse(
            grounded=False,
            reason="The table shows foreign exchange as (2)%, which is unfavorable, not favorable.",
        ),
    ),
    RagVerifierExample(
        documents="""
[Asset: report_a1b2c3d4.pdf] [Section: Summary] [Location: Page 2]

Net earnings increased 2% to $15.0 billion.
""",
        answer="Net earnings were $15.0 billion.",
        response=RagVerifierResponse(
            grounded=True,
            reason="The answer directly matches the value stated in the document.",
        ),
    ),
    RagVerifierExample(
        documents="""
[Asset: report_a1b2c3d4.pdf] [Section: Summary] [Location: Page 2]

Net earnings increased 2% to $15.0 billion.
""",
        answer="Net earnings increased 12% to $15.0 billion.",
        response=RagVerifierResponse(
            grounded=False,
            reason="The document states net earnings increased 2%, not 12%.",
        ),
    ),
    RagVerifierExample(
        documents="""
    [Asset: report_a1b2c3d4.pdf] [Section: Data Table] [Location: Page 5]

    | Segment | Foreign Exchange |
    |---------|------------------|
    | Grooming | (6)%            |
    """,
        answer="Foreign exchange had an unfavorable impact of -6% on Grooming.",
        response=RagVerifierResponse(
            grounded=True,
            reason="The table shows (6)% which corresponds to -6%, matching the answer.",
        ),
    ),
    RagVerifierExample(
        documents="""
    [Asset: report_a1b2c3d4.pdf] [Section: Summary] [Location: Page 3]

    Grooming net sales increased high single digits versus the prior year.
    """,
        answer="I cannot find the exact net sales growth percentage for the Grooming segment. The document only states that growth was in the high single digits.",
        response=RagVerifierResponse(
            grounded=True,
            reason="The answer is grounded because the document only provides high single-digit growth and does not specify an exact percentage.",
        ),
    ),
    RagVerifierExample(
        documents="""
    [Asset: report_a1b2c3d4.pdf] [Section: Segment Results] [Location: Page 5]

    | Segment  | Foreign Exchange | Price | Net Sales Growth |
    |----------|------------------|-------|------------------|
    | Grooming | (5)%             | 8%    | 4%               |
    """,
        answer=(
            "I cannot find the exact net sales growth percentage for Grooming "
            "in the first quarter of 2024 in the provided documents. The documents "
            "only show full-year 2024 versus 2023 Grooming net sales growth of 4%."
        ),
        response=RagVerifierResponse(
            grounded=True,
            reason=(
                "The answer is grounded because the document provides full-year "
                "2024 versus 2023 Grooming net sales growth, but does not provide "
                "a first-quarter 2024 Grooming value."
            ),
        ),
    ),
    RagVerifierExample(
        documents="""
    [Asset: report_a1b2c3d4.pdf] [Section: Segment Results] [Location: Page 5]

    | Segment  | Foreign Exchange | Price | Net Sales Growth |
    |----------|------------------|-------|------------------|
    | Grooming | (5)%             | 8%    | 4%               |
    """,
        answer="Grooming net sales growth was 4% in the first quarter of 2024.",
        response=RagVerifierResponse(
            grounded=False,
            reason=(
                "The document shows Grooming net sales growth of 4% for full-year "
                "2024 versus 2023, but does not support the first-quarter 2024 scope."
            ),
        ),
    )
]
