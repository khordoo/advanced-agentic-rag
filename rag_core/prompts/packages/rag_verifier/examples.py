from pydantic import BaseModel, Field

from rag_core.prompts.packages.rag_verifier.schema import RagVerifierResponse


class RagVerifierExample(BaseModel):
    documents: str = Field(description="The documents")
    answer: str = Field(description="The generated answer to verify")
    response: RagVerifierResponse = Field(description="The expected verifier response")


EXAMPLES = [
    RagVerifierExample(
        documents="""
[Doc: report.pdf] [Section: Data Table] [Location: Page 5]

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
[Doc: report.pdf] [Section: Data Table] [Location: Page 5]

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
[Doc: report.pdf] [Section: Data Table] [Location: Page 5]

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
[Doc: report.pdf] [Section: Data Table] [Location: Page 5]

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
[Doc: report.pdf] [Section: Summary] [Location: Page 2]

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
[Doc: report.pdf] [Section: Summary] [Location: Page 2]

Net earnings increased 2% to $15.0 billion.
""",
        answer="Net earnings increased 12% to $15.0 billion.",
        response=RagVerifierResponse(
            grounded=False,
            reason="The document states net earnings increased 2%, not 12%.",
        ),
    ),
]
