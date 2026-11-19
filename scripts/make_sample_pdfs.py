"""Generate SYNTHETIC banking policy PDFs. All names and figures are fictional."""
from pathlib import Path

import fitz

OUT = Path("data/samples")  # pristine defaults; copied into data/pdfs (the working folder) at startup

DOCS = {
    "lending_requirements": [
        "NORTHBRIDGE BANK - COMMERCIAL LENDING REQUIREMENTS POLICY\n"
        "Document LRP-101 | Version 3.2 | Effective 1 January 2025 | Owner: Credit Policy | SYNTHETIC\n\n"
        "1. Purpose and scope\n\n"
        "This policy sets minimum underwriting requirements for commercial loans originated by the Business Banking "
        "division. It does not apply to consumer lending, credit cards, or mortgage products, which are governed by "
        "separate policies.\n\n"
        "2. Borrower eligibility\n\n"
        "A borrower must have at least 24 months of operating history to qualify for an unsecured commercial loan. "
        "Secured loans may be considered for borrowers with 12 months of operating history. Start-up businesses with "
        "less than 12 months of history are not eligible for credit under this policy. Borrowers must be incorporated "
        "or registered in a jurisdiction where the Bank is licensed to operate.",
        "3. Debt service coverage\n\n"
        "The minimum debt service coverage ratio (DSCR) for commercial loans is 1.25x, calculated on the borrower's "
        "most recent audited financial statements. For loans with a term over 10 years, the minimum DSCR is 1.35x.\n\n"
        "Exceptions to the minimum DSCR require written approval from the Regional Credit Officer and must be recorded "
        "in the credit memo with a documented mitigating factor.\n\n"
        "4. Leverage limits\n\n"
        "Total debt to EBITDA must not exceed 4.0x at origination for general commercial borrowers. For borrowers in "
        "the construction sector the limit is 3.0x because of earnings volatility.",
        "5. Loan-to-value limits\n\n"
        "Maximum loan-to-value (LTV) by collateral type:\n"
        "- Commercial real estate (owner-occupied): 75%\n"
        "- Commercial real estate (investment): 70%\n"
        "- Land loans: 65%\n"
        "- Equipment: 80% of purchase price\n"
        "- Accounts receivable: 85% of eligible receivables aged under 90 days\n\n"
        "Inventory-backed lending is limited to 50% of the lower of cost or market value.\n\n"
        "6. Concentration limits\n\n"
        "Exposure to a single borrower group must not exceed 10% of the Bank's Tier 1 capital. Exposure to any one "
        "industry sector must not exceed 25% of the total commercial loan portfolio.",
        "7. Approval authority\n\n"
        "Loans above USD 5,000,000 require Credit Committee approval. Loans up to USD 5,000,000 may be approved by a "
        "Senior Credit Officer. Loans up to USD 500,000 that meet all standard criteria may be approved by a Credit "
        "Officer.\n\n"
        "No officer may approve a loan to a related party. Related-party loans always require Credit Committee approval "
        "regardless of amount, plus a recorded vote by the Board Risk Committee.\n\n"
        "8. Policy exceptions\n\n"
        "Any exception to sections 2 through 6 must be logged in the Exceptions Register within 5 business days of "
        "approval. Exceptions are reported to the Credit Committee quarterly.",
    ],
    "credit_risk_review": [
        "NORTHBRIDGE BANK - CREDIT RISK REVIEW PROCEDURE\n"
        "Document CRR-204 | Version 2.1 | Effective 1 March 2025 | Owner: Credit Risk Management | SYNTHETIC\n\n"
        "1. Purpose\n\n"
        "This procedure defines how commercial loans are risk-rated and how often they are reviewed, so deteriorating "
        "credits are identified early.\n\n"
        "2. Risk rating scale\n\n"
        "Loans are rated Pass, Watch, Substandard, Doubtful, or Loss. Ratings are assigned by the originating officer "
        "and independently validated by Credit Risk Management for all loans above USD 1,000,000.\n\n"
        "3. Review frequency\n\n"
        "Loans rated Pass are reviewed annually. Loans rated Watch are reviewed every 6 months. Loans rated "
        "Substandard or worse are reviewed quarterly. Any loan above USD 10,000,000 is reviewed at least semi-annually "
        "regardless of rating.",
        "4. Downgrade triggers\n\n"
        "A loan must be downgraded to Watch when any of the following occur:\n"
        "- DSCR falls below 1.10x\n"
        "- Payments are 30 days past due\n"
        "- A financial covenant is breached and not cured within 30 days\n"
        "- Audited financials are not delivered within 120 days of fiscal year end\n\n"
        "A loan must be downgraded to Substandard when payments are 90 days past due or when DSCR falls below 1.00x.",
        "5. Covenant monitoring\n\n"
        "Financial covenant compliance certificates must be collected within 45 days of each quarter end. Missed "
        "certificates are escalated to the Relationship Manager after 10 business days and to the Regional Credit "
        "Officer after 30 business days.\n\n"
        "6. Provisioning\n\n"
        "Loans rated Substandard require a specific provision of at least 15% of the outstanding balance, net of "
        "eligible collateral. Doubtful loans require at least 50%. Loss-rated loans must be fully provided for and "
        "written off within 90 days.",
        "7. Escalation and reporting\n\n"
        "Credit Risk Management reports all Watch and worse loans to the Credit Committee monthly. A single borrower "
        "downgrade to Substandard with exposure above USD 2,000,000 must be reported to the Chief Risk Officer within "
        "5 business days.\n\n"
        "8. Independent validation\n\n"
        "At least 10% of Pass-rated loans, selected at random, are re-rated each year by Credit Risk Management to test "
        "the accuracy of originating-officer ratings. A rating disagreement rate above 15% triggers a review of the "
        "originating team.",
    ],
    "loan_documentation": [
        "NORTHBRIDGE BANK - LOAN DOCUMENTATION STANDARDS\n"
        "Document LDS-310 | Version 1.8 | Effective 1 February 2025 | Owner: Credit Operations | SYNTHETIC\n\n"
        "1. Purpose\n\n"
        "This standard lists the documents required in every commercial loan file before funds are disbursed.\n\n"
        "2. Required documents\n\n"
        "Every commercial loan file must contain: a signed credit agreement, the borrower's last three years of "
        "financial statements, a completed KYC checklist, a board resolution authorising the borrowing, and evidence "
        "of collateral perfection where applicable.\n\n"
        "Personal guarantees, where required, must be accompanied by a personal financial statement dated within "
        "12 months.",
        "3. KYC and beneficial ownership\n\n"
        "The KYC checklist must identify every individual who owns 25% or more of the borrower. KYC files must be "
        "refreshed every 3 years for standard-risk borrowers and annually for high-risk borrowers. Loans must not be "
        "disbursed while KYC is incomplete or expired.\n\n"
        "4. Document retention\n\n"
        "Loan files are retained for 7 years after loan maturity or payoff. Records subject to a legal hold must be "
        "retained until the hold is released by the Legal department.",
        "5. Collateral documentation\n\n"
        "Real estate collateral requires an appraisal dated within 12 months of the loan approval date. Appraisals "
        "must be performed by an approved third-party appraiser. Equipment collateral requires an invoice or "
        "independent valuation and a filed UCC-1 financing statement.\n\n"
        "6. Insurance\n\n"
        "Property collateral must be insured for at least its replacement value, with the Bank named as loss payee. "
        "Evidence of insurance must be renewed annually.",
        "7. Pre-disbursement checklist\n\n"
        "Credit Operations must confirm, and a second officer must independently verify, that all required documents "
        "are present, signed, and dated before any disbursement. Missing documents may be waived only by the Head of "
        "Credit Operations, and a waiver expires after 30 days unless the document is received.\n\n"
        "8. Amendments\n\n"
        "Any amendment to a signed credit agreement must be documented in a written amendment signed by all parties "
        "and filed in the loan file within 10 business days.",
    ],
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, pages in DOCS.items():
        doc = fitz.open()
        for i, text in enumerate(pages, 1):
            page = doc.new_page()
            page.insert_textbox(fitz.Rect(54, 54, 558, 760), text, fontsize=11, fontname="helv")
            page.insert_text((54, 810), f"Page {i} of {len(pages)} - SYNTHETIC DOCUMENT", fontsize=8, fontname="helv")
        doc.save(OUT / f"{name}.pdf")
        print("wrote", OUT / f"{name}.pdf")
