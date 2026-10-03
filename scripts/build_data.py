"""Generate data/corpus.json and data/qa.json (fully synthetic, fictional insurer)."""
import json
from pathlib import Path

DOCS = [
 ("d01", "Overview of Lenders Mortgage Insurance", [
  ("What LMI is", "Lenders mortgage insurance (LMI) protects the lender, not the borrower, if a home loan defaults and the sale of the property does not cover the outstanding balance. The borrower pays the premium, usually once, and it is often added to the loan amount."),
  ("When LMI applies", "Northwind Lenders requires LMI when the loan to value ratio (LVR) is above 80 percent. A borrower with a 10 percent deposit will therefore be asked to pay LMI, while a borrower with a 25 percent deposit will not."),
  ("Who is covered", "The policy covers the lender named on the loan contract. Co-borrowers and guarantors are not insured parties. If the borrower refinances with another lender, a new LMI policy may be required by that lender."),
  ("Policy term", "The policy starts on the day the loan is settled and ends when the loan is fully repaid or discharged. There is no need to renew it and the premium does not change during the life of the loan."),
  ("Why it exists", "LMI lets lenders approve loans with smaller deposits. This helps first home buyers enter the market sooner, in exchange for a one-off cost that depends on loan size, deposit and borrower profile."),
 ]),
 ("d02", "Eligibility and Premium Calculation", [
  ("Eligibility criteria", "To be eligible, applicants must be at least 18, hold an income that meets serviceability checks, and provide at least a 5 percent genuine savings deposit held for three months. Loans above 95 percent LVR are not accepted."),
  ("Premium factors", "The premium is calculated from three factors: the LVR, the loan amount, and whether the property is an owner-occupied home or an investment. Higher LVR and larger loans increase the premium."),
  ("Premium bands", "For an owner-occupied loan of 300,000 dollars, the example premium is about 1.0 percent of the loan at 85 percent LVR, 1.8 percent at 90 percent LVR and 3.2 percent at 95 percent LVR. These figures are illustrative only."),
  ("Capitalising the premium", "Borrowers can pay the premium upfront at settlement or capitalise it, meaning it is added to the loan balance. Capitalising increases total interest paid because the premium is repaid over the loan term."),
  ("Stamp duty on premium", "State governments may charge stamp duty on the LMI premium. The duty is calculated by the state revenue office, shown on the settlement statement, and is not refundable by Northwind."),
 ]),
 ("d03", "Claims Process", [
  ("When a claim arises", "A claim arises only after the lender has sold the secured property and the sale proceeds are less than the debt owed. The lender, not the borrower, lodges the claim with the insurer."),
  ("Lodging a claim", "The lender must lodge the claim within 60 days of the sale settlement date, supplying the loan contract, arrears history, valuation reports and the sale contract. Late claims may be reduced or declined."),
  ("Assessment time", "Northwind aims to assess complete claims within 30 business days. If extra documents are needed, the assessment clock pauses until the lender provides them."),
  ("Claim payment", "The insurer pays the lender the verified shortfall, up to the policy limit, which is the original loan amount plus capitalised premium. Interest and reasonable sale costs may also be included."),
  ("Recovery from borrower", "After paying a claim, the insurer may seek to recover the shortfall from the borrower. This is called subrogation. The borrower remains legally liable for the debt even though the lender has been paid."),
 ]),
 ("d04", "Exclusions and Limitations", [
  ("Fraud and misrepresentation", "No claim is payable if the loan was obtained through fraud or material misrepresentation, such as falsified payslips or undisclosed liabilities. The insurer may also cancel the policy and recover amounts already paid."),
  ("Lender breaches", "Claims can be declined where the lender did not follow its own lending policy, did not complete required checks, or failed to act on early signs of arrears in a reasonable time."),
  ("Property damage", "Losses caused by uninsured physical damage to the property are excluded. Borrowers must keep building insurance in place, and lenders should confirm that the cover is current before lodging a claim."),
  ("Policy limit", "The insurer will never pay more than the policy limit. Amounts above the limit, such as extra legal costs beyond what is reasonable, remain the lender's responsibility."),
  ("Excluded loan types", "Commercial loans, construction loans without a completion plan, and loans to non-residents without verified income are not covered by this policy."),
 ]),
 ("d05", "Financial Hardship and Arrears", [
  ("Hardship support", "Borrowers facing financial hardship should contact their lender as early as possible. Lenders can offer repayment pauses, reduced repayments for a set period, or a term extension to avoid default."),
  ("Arrears notices", "If repayments are missed, the lender sends a notice after 14 days, a second notice after 30 days and a formal default notice after 60 days. Each notice states the amount owed and how to arrange support."),
  ("Insurer involvement", "The insurer is notified when a loan reaches 90 days in arrears. At that point the insurer may work with the lender on a recovery plan, but the borrower still deals only with the lender."),
  ("Selling voluntarily", "Borrowers may choose to sell the property themselves to reduce losses. A voluntary sale often achieves a better price than a forced sale and can lower the shortfall that any claim would cover."),
  ("Impact on credit", "Missed repayments are reported to credit bureaus and can lower a credit score for up to five years, which may make future borrowing harder or more expensive."),
 ]),
 ("d06", "Cancellation and Premium Refunds", [
  ("Cancelling the policy", "Because the policy protects the lender, a borrower cannot cancel it directly. The policy ends only when the loan is repaid, discharged or refinanced and the lender notifies Northwind."),
  ("Refund eligibility", "A partial premium refund may be available if the loan is fully repaid within the first 12 months. After 12 months no refund is payable under the standard policy."),
  ("Refund calculation", "Within the first 12 months the refund equals the premium less an administration fee of 150 dollars, reduced pro rata for each full month the policy has been in force."),
  ("Requesting a refund", "Borrowers request refunds through their lender, who submits the payoff statement and policy details to Northwind. Refunds are paid to the lender, who passes them on to the borrower."),
  ("Refinancing and transfers", "When a loan is refinanced with the same lender, the policy may be transferred without a new premium if the loan to value ratio has not increased."),
 ]),
 ("d07", "Privacy and Data Handling", [
  ("Information we collect", "Northwind collects the personal and financial information supplied by the lender, including identity details, income evidence, property valuations and repayment history, only to assess risk and manage claims."),
  ("How data is used", "Data is used to price policies, assess claims, prevent fraud and meet regulatory obligations. It is not sold, and it is not used for marketing unless the borrower gives separate consent."),
  ("Retention period", "Records are kept for seven years after the policy ends so that late claims and regulatory reviews can be handled. After that period data is securely deleted or de-identified."),
  ("Access and correction", "Individuals may ask to see the information held about them, and request corrections if it is inaccurate. Requests are answered within 30 days and are free of charge."),
  ("Overseas disclosure", "Some processing, such as document storage, may be carried out by service providers in other countries. Northwind requires those providers to meet equivalent privacy and security standards."),
 ]),
 ("d08", "Complaints and Dispute Resolution", [
  ("Making a complaint", "Complaints can be made by phone, email or the online form. Northwind acknowledges each complaint within one business day and records it in a central register."),
  ("Resolution timeframes", "Standard complaints are resolved within 30 calendar days. Complex complaints, such as those involving a declined claim, may take longer, and the complainant is updated at least every 10 days."),
  ("Internal review", "If the complainant disagrees with the outcome, they may ask for an internal review by a senior officer who was not involved in the original decision. The review is completed within 15 business days."),
  ("External dispute resolution", "If the complaint is still unresolved, the complainant can take it to the Australian Financial Complaints Authority, a free and independent service, without needing a lawyer."),
  ("Compensation", "Where Northwind has made an error that caused a financial loss, it will correct the error and may offer compensation reasonable for the loss, such as refunding an incorrect fee."),
 ]),
]

# (question, gold section "dXX-sN" (1-indexed), reference answer)
QA = [
 ("Who benefits from lenders mortgage insurance, the borrower or the bank?", "d01-s1", "It protects the lender, not the borrower."),
 ("At what deposit size will I have to pay LMI?", "d01-s2", "LMI is required when the LVR is above 80 percent."),
 ("Does the policy need to be renewed every year?", "d01-s4", "No, it runs until the loan is repaid with no renewal."),
 ("Why do lenders offer insurance that allows smaller deposits?", "d01-s5", "LMI lets lenders approve loans with smaller deposits."),
 ("What is the minimum savings deposit and how long must it be held?", "d02-s1", "At least 5 percent genuine savings held for three months."),
 ("Which factors drive the cost of my premium?", "d02-s2", "LVR, loan amount and owner-occupied versus investment."),
 ("How much would a 90 percent LVR loan cost in premium?", "d02-s3", "About 1.8 percent of the loan."),
 ("What happens if I add the premium to my loan balance?", "d02-s4", "Capitalising increases total interest paid."),
 ("Is government duty charged on the premium and can it be refunded?", "d02-s5", "State stamp duty may apply and is not refundable by Northwind."),
 ("Who submits the claim after a property is sold at a loss?", "d03-s1", "The lender lodges the claim with the insurer."),
 ("How many days does the lender have to file the claim after settlement?", "d03-s2", "Within 60 days of the sale settlement date."),
 ("How long does Northwind take to assess a complete claim?", "d03-s3", "About 30 business days."),
 ("What is the most the insurer will pay out on a claim?", "d03-s4", "Up to the policy limit, the original loan plus capitalised premium."),
 ("Can the insurer chase me for money after paying the bank?", "d03-s5", "Yes, through subrogation, the borrower stays liable for the debt."),
 ("Will a claim be paid if I gave false income documents?", "d04-s1", "No, fraud and misrepresentation make a claim not payable."),
 ("What if the bank ignored its own lending rules?", "d04-s2", "Claims can be declined for lender breaches of policy."),
 ("Do I need building insurance for the property to be covered?", "d04-s3", "Yes, uninsured property damage is excluded."),
 ("Are construction loans and commercial loans covered?", "d04-s5", "No, commercial and certain construction loans are excluded."),
 ("What help is available if I can no longer afford repayments?", "d05-s1", "Contact the lender for pauses, reduced repayments or term extension."),
 ("When does the lender send a formal default notice?", "d05-s2", "After 60 days of missed repayments."),
 ("At how many days overdue does the insurer get told?", "d05-s3", "At 90 days in arrears."),
 ("Is selling the home myself better than a forced sale?", "d05-s4", "A voluntary sale often achieves a better price and lowers the shortfall."),
 ("How long do missed payments stay on my credit file?", "d05-s5", "Up to five years."),
 ("Can I cancel the policy myself to save money?", "d06-s1", "No, borrowers cannot cancel it directly."),
 ("Can I get some premium back if I pay off the loan quickly?", "d06-s2", "A partial refund is possible if repaid within 12 months."),
 ("What fee is deducted from a premium refund?", "d06-s3", "An administration fee of 150 dollars."),
 ("Can the policy move with me if I refinance with the same bank?", "d06-s5", "It may be transferred without a new premium if LVR has not increased."),
 ("Does Northwind sell my information to marketers?", "d07-s2", "No, data is not sold or used for marketing without consent."),
 ("How many years are my records kept after the policy ends?", "d07-s3", "Seven years."),
 ("Can I ask to see the personal data you hold on me?", "d07-s4", "Yes, requests are answered within 30 days, free of charge."),
 ("How quickly will a complaint be acknowledged?", "d08-s1", "Within one business day."),
 ("Where can I escalate if I am still unhappy after the internal review?", "d08-s4", "The Australian Financial Complaints Authority."),
]

# Out-of-scope questions the handbook cannot answer: a grounded system should abstain.
UNANSWERABLE = [
 "What is the current interest rate on a 30 year fixed home loan?",
 "Does Northwind offer home contents insurance for renters?",
 "How do I change the bank account used for my direct debit repayments?",
 "Who is the chief executive officer of Northwind Lenders?",
 "Are there premium discounts for first home buyers?",
 "How much capital gains tax do I pay when selling an investment property?",
 "Can I claim the insurance premium as a tax deduction?",
 "What is the weather like in Sydney during December?",
 "How do I apply for a personal car loan?",
 "Which suburbs have had the best property price growth this year?",
]


def main():
    corpus = []
    for did, title, secs in DOCS:
        corpus.append({"id": did, "title": title, "sections": [
            {"id": f"{did}-s{i+1}", "heading": h, "text": t} for i, (h, t) in enumerate(secs)]})
    qa = [{"id": f"q{i+1:02d}", "question": q, "gold": g, "answer": a} for i, (q, g, a) in enumerate(QA)]
    ids = {s["id"] for d in corpus for s in d["sections"]}
    assert all(x["gold"] in ids for x in qa), "gold id missing"
    out = Path(__file__).resolve().parents[1] / "data"
    (out / "corpus.json").write_text(json.dumps({"synthetic": True, "documents": corpus}, indent=1))
    (out / "qa.json").write_text(json.dumps(qa, indent=1))
    (out / "qa_unanswerable.json").write_text(json.dumps([{"id": f"u{i+1:02d}", "question": q} for i, q in enumerate(UNANSWERABLE)], indent=1))
    (out / "pricing.json").write_text(json.dumps({
        "note": "Illustrative placeholder prices (USD per 1M tokens). NOT real vendor prices; edit to match your provider.",
        "models": {"small": {"in": 0.25, "out": 1.25}, "medium": {"in": 3.0, "out": 15.0}, "large": {"in": 15.0, "out": 75.0}},
        "system_tokens": 60, "answer_tokens": 80}, indent=1))
    print(len(corpus), "docs,", len(ids), "sections,", len(qa), "questions,", len(UNANSWERABLE), "unanswerable")

if __name__ == "__main__":
    main()
