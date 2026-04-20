from rag_core.prompts.specs.base import RagFinalResponseSchema , Citation


EXAMPLES = [
    # Case 1: Standard / Single Citation (Concise)
    RagFinalResponseSchema(
        answer="Net sales for the Fabric Care segment increased by 4% in the third quarter, primarily driven by higher pricing and favorable product mix [1].",
        citations=[
            Citation(
                document_index=1, file_name="Q3_Earnings_Report.pdf", page_number=8
            )
        ],
    ),
    # Case 2: Multi-Citation / Long Answer (Detailed Synthesis)
    RagFinalResponseSchema(
        answer="The three specific macroeconomic or programmatic factors that caused the estimated fair value of the Gillette indefinite-lived intangible asset to drop, resulting in the $1.3 billion impairment, were a higher discount rate, weakening of several currencies relative to the U.S. dollar, and the impact of the limited market portfolio restructuring program [1]. The specific percentage impact of foreign exchange on the net sales growth for the Grooming segment was -5% [2].",
        citations=[
            Citation(document_index=1, file_name="report.pdf", page_number=4),
            Citation(document_index=2, file_name="report.pdf", page_number=5),
        ],
    ),
    # Case 3: Negative / Null Case (Graceful Fallback)
    RagFinalResponseSchema(
        answer="I cannot find the answer in the provided documents.", citations=[]
    ),
]
