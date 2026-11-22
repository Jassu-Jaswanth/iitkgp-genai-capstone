# report/: the 12-page technical report (LaTeX)

**Read it:** [report.pdf](report.pdf) (rebuilt and committed whenever someone changes the text).

## Add your part
1. Open your section in `sections/` and write. One file per section, so people don't clash:

   | File | Section | Main owner |
   |---|---|---|
   | `00-abstract.tex` | Abstract | Jaswanth |
   | `01-introduction.tex` | Introduction and task spec (what it does / refuses / a correct answer) | Jaswanth |
   | `02-data.tex` | Data and evaluation sets | Jaswanth, Vijay |
   | `03-prompting.tex` | Prompt baseline and how the 8B model breaks | Jaswanth |
   | `04-retrieval.tex` | Chunking, hybrid search, reranking, recall@k | Rajesh |
   | `05-agent.tex` | Agent graph, tools, critic, error handling | Jaswanth |
   | `06-evaluation.tex` | Metrics, failure taxonomy, tracing | Jaswanth |
   | `07-small-model.tex` | What we did to compensate for a small model (graded section) | everyone |
   | `08-deployment.tex` | App, containers, rate limiting, caching | Jaswanth |
   | `09-responsible-ai.tex` | Prompt injection, PII, refusals, human-in-the-loop | open |
   | `10-limitations.tex` | Limitations and future work | everyone |
   | `11-team.tex` | Who did what | everyone (add yourself) |

2. **Something not done yet?** Leave a red placeholder so it shows in the PDF:
   `\todo[Your Name]{what is missing}` (a box), or `\todoin[Your Name]{...}` inside a table cell.
   Leave the name out if nobody owns it yet. Replace it with real text when done.
3. **Sources:** add a `\bibitem{key}` to `references.tex` and cite it with `\cite{key}`.
4. **Names / title:** the title block is at the top of `main.tex`.

## Build the PDF
```bash
./build.sh            # -> report.pdf, prints the page count
```
Uses TeX Live if you have it (`latexmk`), otherwise Docker (`texlive/texlive:latest-medium`, ~2 GB the first
time; nothing to install). Then commit `report.pdf` together with your `.tex` changes.

## Rules
- **12 pages maximum** (the build prints the count). Tables and figures count.
- Numbers must come from files in the repo (`eval/results/`, `docs/DATASET.md`), so anyone can check them.
- Keep the voice plain: what we did, what happened, what we learned.
