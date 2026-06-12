"""CLI: python -m agent.advisor [--asof YYYY-MM-DD] [--notify] [--dry-run] [--no-llm]"""
import argparse
import sys

from agent.advisor import run_advisor


def main():
    ap = argparse.ArgumentParser(prog="agent.advisor",
                                 description="NOX AI portföy danışmanı")
    ap.add_argument("--asof", default=None, help="Değerlendirme günü (vars: bugün TR)")
    ap.add_argument("--notify", action="store_true", help="Telegram'a gönder")
    ap.add_argument("--dry-run", action="store_true",
                    help="LLM + Telegram yok; pack & fallback raporu konsola")
    ap.add_argument("--no-llm", action="store_true",
                    help="LLM'i atla, deterministik fallback kullan")
    ap.add_argument("--portfolio-path", default=None,
                    help="Lokal portföy JSON yolu (vars: env/GitHub/agent/portfolio.json)")
    ap.add_argument("--no-publish", action="store_true",
                    help="advisory_latest.json'ı private repoya yazma")
    a = ap.parse_args()

    advisory = run_advisor(asof=a.asof, notify=a.notify, dry_run=a.dry_run,
                           use_llm=not a.no_llm, portfolio_path=a.portfolio_path,
                           publish_latest=not a.no_publish)
    # bug değil veri-koşulu ayrımı: rapor üretilebildiyse exit 0
    return 0 if advisory else 1


if __name__ == "__main__":
    sys.exit(main())
