# Cookies

This directory held session cookies for the six sock-puppet accounts used in
the audit (one fresh + one light-seeded per platform; see thesis Section 3.4
"Sock-puppet accounts" and Section 3.5 "Per-hour collection protocol").

For submission, the live cookies have been removed — they are session
credentials and must not be redistributed.

## Expected file structure

To re-run the scrapers, populate this directory with cookies exported from
authenticated browser sessions of the following accounts:

| Platform  | Persona            | Filename                          |
|-----------|--------------------|-----------------------------------|
| TikTok    | Sandra (fresh)     | `tiktok_sandra_fresh.json`        |
| TikTok    | Laura (seeded)     | `tiktok_laura_seed.json`          |
| Instagram | Sandra (fresh)     | `insta_sandra_fresh.json`         |
| Instagram | Andrea (seeded)    | `insta_andrea_seeded.json`        |
| LinkedIn  | Melanie (fresh)    | `linkedin_melanie_fresh.json`     |
| LinkedIn  | Simone (seeded)    | `linkedin_simone_seeded.json`     |

Each file is a JSON array of cookie objects in the standard
EditThisCookie / Cookie-Editor browser-extension export format. The
LinkedIn variants must include the nine essential cookies listed in
`02_scraper/01_core/auth.py` (`li_at`, `JSESSIONID`, `bcookie`,
`bscookie`, `li_rm`, `liap`, `li_gc`, `li_mc`, `lidc`).

For analysis-only replication (re-running the modelling pipeline against
the supplied `04_database/scraper.db`), no cookies are required.
