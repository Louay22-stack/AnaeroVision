# ============================================================
# ANAEROVISION — AI-Powered Anaerobic Digestion Intelligence
# Startup-style UI/UX redesign — navy/teal brand system
# ============================================================

import html
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

LOGO_SVG = '<svg class="av-logo-svg" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="AnaeroVision logo">\n  <defs>\n    <linearGradient id="avLogoGradient" x1="8" y1="8" x2="56" y2="56" gradientUnits="userSpaceOnUse">\n      <stop stop-color="#7ECECA"/>\n      <stop offset="0.48" stop-color="#2E7D7B"/>\n      <stop offset="1" stop-color="#1C2E4A"/>\n    </linearGradient>\n    <linearGradient id="avGlow" x1="20" y1="18" x2="44" y2="46" gradientUnits="userSpaceOnUse">\n      <stop stop-color="#FFFFFF" stop-opacity="0.96"/>\n      <stop offset="1" stop-color="#DFF8F5" stop-opacity="0.72"/>\n    </linearGradient>\n  </defs>\n  <rect x="2" y="2" width="60" height="60" rx="17" fill="url(#avLogoGradient)"/>\n  <path d="M15 42.5C19.7 37.2 23.4 30.9 26.5 23.6C28 20.1 30 18.3 32.4 18.3C34.7 18.3 36.8 20.1 38.3 23.6C41.3 30.7 44.8 37 49 42.5" fill="none" stroke="url(#avGlow)" stroke-width="5.2" stroke-linecap="round"/>\n  <path d="M21 38.5H43.5" stroke="#FFFFFF" stroke-width="4.4" stroke-linecap="round"/>\n  <circle cx="47.8" cy="17.3" r="3.2" fill="#A7ECE7"/>\n  <circle cx="51.5" cy="25.3" r="1.8" fill="#FFFFFF" fill-opacity="0.88"/>\n</svg>'

# Encode the AnaeroVision mark as a data URI so it can be used as the browser
# tab (favicon) icon — Streamlit's page_icon accepts an emoji, local image
# path, or an image URL (a base64 data URI works as an image URL).
import base64
LOGO_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode(LOGO_SVG.encode("utf-8")).decode("ascii")

# ------------------------------------------------------------------
# Icon set — small line icons (stroke=currentColor) replacing text
# placeholders like "AI" / "OK" / "RUN" so icon slots read as a real,
# consistent iconography system rather than typographic stand-ins.
# ------------------------------------------------------------------
_ICON_BASE = 'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'

ICON_WORKFLOW = f'<svg {_ICON_BASE}><circle cx="5" cy="6" r="2.4"/><circle cx="19" cy="6" r="2.4"/><circle cx="12" cy="18" r="2.4"/><path d="M7.1 7.4 10 15.6M16.9 7.4 14 15.6"/></svg>'
ICON_CHIP = f'<svg {_ICON_BASE}><rect x="7" y="7" width="10" height="10" rx="1.6"/><path d="M9.5 7V3.6M14.5 7V3.6M9.5 21v-3.4M14.5 21v-3.4M7 9.5H3.6M7 14.5H3.6M21 9.5h-3.4M21 14.5h-3.4"/></svg>'
ICON_FLASK = f'<svg {_ICON_BASE}><path d="M10 3h4M10 3v5.2L4.8 17.4A2 2 0 0 0 6.6 20.4h10.8a2 2 0 0 0 1.8-3L14 8.2V3"/><path d="M8.5 14.5h7"/></svg>'
ICON_CHECK = f'<svg {_ICON_BASE}><circle cx="12" cy="12" r="9"/><path d="M8.2 12.3 10.7 14.8 15.8 9.4"/></svg>'
ICON_BULB = f'<svg {_ICON_BASE}><path d="M9 18h6M10 21h4M8 14.5A5 5 0 1 1 16 14.5c-.9.9-1.4 1.7-1.5 2.7H9.5C9.4 16.2 8.9 15.4 8 14.5Z"/></svg>'
ICON_PLAY = f'<svg {_ICON_BASE}><circle cx="12" cy="12" r="9"/><path d="M10.2 8.6 15.4 12l-5.2 3.4Z" fill="currentColor"/></svg>'

st.set_page_config(
    page_title="AnaeroVision — AD Process Intelligence",
    page_icon=LOGO_DATA_URI,
    layout="wide",
    initial_sidebar_state="expanded",
)
HERO_VISUAL = r'''<div class="bio-visual" aria-hidden="true">
  <div class="bio-orbit bio-orbit-one"></div>
  <div class="bio-orbit bio-orbit-two"></div>
  <div class="bio-panel">
    <div class="bio-panel-head">
      <span>Bioprocess optimization layer</span>
      <span class="live-dot"><i></i> Models ready</span>
    </div>
    <div class="bio-process-map">
      <div class="bio-inputs">
        <div class="bio-node">
          <span class="bio-node-code">IN 01</span>
          <b>Feedstock profile</b>
          <small>Composition & substrate class</small>
        </div>
        <div class="bio-node">
          <span class="bio-node-code">IN 02</span>
          <b>Operating window</b>
          <small>T · pH · HRT · OLR · ISR</small>
        </div>
      </div>
      <div class="reactor-wrap">
        <svg class="reactor-svg" viewBox="0 0 220 270" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="vesselGlass" x1="44" y1="35" x2="175" y2="238" gradientUnits="userSpaceOnUse">
              <stop stop-color="#FFFFFF" stop-opacity="0.16"/>
              <stop offset="1" stop-color="#7ECECA" stop-opacity="0.05"/>
            </linearGradient>
            <linearGradient id="brothFill" x1="72" y1="118" x2="151" y2="231" gradientUnits="userSpaceOnUse">
              <stop stop-color="#7ECECA" stop-opacity="0.78"/>
              <stop offset="1" stop-color="#2E7D7B" stop-opacity="0.56"/>
            </linearGradient>
            <filter id="reactorGlow" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="5"/>
            </filter>
          </defs>
          <path d="M33 93H64" stroke="#7ECECA" stroke-width="2" stroke-linecap="round"/>
          <path d="M56 87L64 93L56 99" fill="none" stroke="#7ECECA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M157 72H190V45" stroke="#A6E0DC" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M184 53L190 45L196 53" fill="none" stroke="#A6E0DC" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <rect x="68" y="34" width="88" height="22" rx="8" fill="#183650" stroke="#77C9C4" stroke-opacity="0.55"/>
          <path d="M78 54V70" stroke="#83D4CF" stroke-width="3" stroke-linecap="round"/>
          <path d="M146 54V70" stroke="#83D4CF" stroke-width="3" stroke-linecap="round"/>
          <path d="M67 69H157V199C157 223 137 243 112 243C87 243 67 223 67 199V69Z" fill="url(#vesselGlass)" stroke="#9ADDD9" stroke-width="2.2"/>
          <path d="M72 124C88 116 98 132 112 124C126 116 138 132 152 124V199C152 220 134 237 112 237C90 237 72 220 72 199V124Z" fill="url(#brothFill)"/>
          <path d="M72 124C88 116 98 132 112 124C126 116 138 132 152 124" fill="none" stroke="#C7F3F0" stroke-width="2" stroke-opacity="0.82"/>
          <path d="M112 57V199" stroke="#D9F5F2" stroke-width="3" stroke-linecap="round"/>
          <path d="M91 145H133" stroke="#D9F5F2" stroke-width="4" stroke-linecap="round"/>
          <path d="M97 170H127" stroke="#D9F5F2" stroke-width="4" stroke-linecap="round"/>
          <circle class="bio-bubble" style="animation-delay:0s" cx="91" cy="186" r="5" fill="#DFF8F5" fill-opacity="0.88"/>
          <circle class="bio-bubble" style="animation-delay:.7s" cx="130" cy="201" r="4" fill="#C7F3F0" fill-opacity="0.82"/>
          <circle class="bio-bubble" style="animation-delay:1.4s" cx="102" cy="215" r="3" fill="#FFFFFF" fill-opacity="0.86"/>
          <circle class="bio-bubble" style="animation-delay:.35s" cx="137" cy="157" r="3.5" fill="#FFFFFF" fill-opacity="0.78"/>
          <circle class="bio-bubble" style="animation-delay:1.1s" cx="84" cy="160" r="2.5" fill="#DFF8F5" fill-opacity="0.72"/>
          <circle class="bio-glow-dot" cx="76" cy="109" r="5" fill="#7ECECA" filter="url(#reactorGlow)"/>
          <circle class="bio-glow-dot" style="animation-delay:.8s" cx="76" cy="109" r="3.5" fill="#DFF8F5"/>
          <circle class="bio-glow-dot" style="animation-delay:.4s" cx="151" cy="92" r="5" fill="#7ECECA" filter="url(#reactorGlow)"/>
          <circle class="bio-glow-dot" style="animation-delay:1.2s" cx="151" cy="92" r="3.5" fill="#DFF8F5"/>
          <path d="M88 243V254M136 243V254M80 255H144" stroke="#9ADDD9" stroke-width="2.2" stroke-linecap="round"/>
          <text x="112" y="103" text-anchor="middle" fill="#CBEFEB" font-family="IBM Plex Mono, monospace" font-size="9" font-weight="600">BIOREACTOR</text>
        </svg>
        <div class="reactor-caption"><b>Biological conversion</b><span>Monitored reactor state</span></div>
      </div>
      <div class="bio-outputs">
        <div class="bio-node output"><span class="bio-dot"></span><div><b>CH₄ performance</b><small>Yield prediction</small></div></div>
        <div class="bio-node output"><span class="bio-dot"></span><div><b>Process stability</b><small>Inhibition screening</small></div></div>
        <div class="bio-node output warm"><span class="bio-dot"></span><div><b>Failure mechanism</b><small>Root-cause diagnosis</small></div></div>
      </div>
    </div>
    <div class="bio-panel-footer">
      <span>Biology × process engineering × AI</span>
      <span>Decision-support workflow</span>
    </div>
  </div>
</div>'''

CSS = '<style>\n@import url(\'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap\');\n\n:root {\n  --navy-950:#101D31;\n  --navy-900:#1C2E4A;\n  --navy-800:#29415F;\n  --teal-700:#256B69;\n  --teal-600:#2E7D7B;\n  --teal-500:#4AA8A5;\n  --aqua-300:#7ECECA;\n  --aqua-100:#DFF5F3;\n  --paper:#F4F7F8;\n  --surface:#FFFFFF;\n  --surface-2:#EEF3F5;\n  --line:#D8E1E7;\n  --ink:#17283F;\n  --muted:#657489;\n  --faint:#8E9AAB;\n  --safe:#26705A;\n  --warn:#A96B1C;\n  --danger:#963B3B;\n  --shadow:0 18px 50px rgba(28,46,74,.09);\n  --shadow-soft:0 8px 24px rgba(28,46,74,.07);\n}\n\nhtml, body, [class*="css"] { font-family:\'DM Sans\',sans-serif; }\nbody { background:var(--paper); }\n.stApp {\n  color:var(--ink);\n  background:\n    radial-gradient(circle at 82% 0%, rgba(126,206,202,.18), transparent 24rem),\n    linear-gradient(180deg,#F8FAFB 0%,#F2F6F7 48%,#F7F9FA 100%);\n}\n#MainMenu, footer, header { visibility:hidden; }\n[data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }\n.block-container { max-width:1240px; padding:1.1rem 2.1rem 4rem !important; }\n\n/* Sidebar */\n[data-testid="stSidebar"] {\n  width:292px !important;\n  background:\n    radial-gradient(circle at 25% 5%, rgba(126,206,202,.14), transparent 17rem),\n    linear-gradient(180deg,var(--navy-950),var(--navy-900));\n  border-right:1px solid rgba(255,255,255,.06);\n}\n[data-testid="stSidebar"] > div:first-child { padding-top:.4rem; }\n[data-testid="stSidebar"] * { color:#D8E4EC; }\n[data-testid="stSidebar"] h3 {\n  font-family:\'Manrope\',sans-serif !important;\n  color:#8FCFCC !important; font-size:.65rem !important; letter-spacing:.18em;\n  text-transform:uppercase; margin:1.45rem 0 .65rem;\n}\n.sidebar-brand { display:flex; align-items:center; gap:.72rem; padding:.65rem .1rem 1.15rem; border-bottom:1px solid rgba(255,255,255,.08); }\n.sidebar-brand .av-logo-svg { width:42px; height:42px; flex:none; }\n.sidebar-wordmark { font:800 1rem/1.1 \'Manrope\',sans-serif; letter-spacing:-.03em; color:#FFF; }\n.sidebar-wordmark span { color:#8ED7D2; }\n.sidebar-sub { color:#71869C; font-size:.67rem; margin-top:.25rem; }\n.sidebar-status { display:flex; align-items:center; justify-content:space-between; margin:1rem 0 1.25rem; padding:.72rem .8rem; border:1px solid rgba(126,206,202,.16); border-radius:12px; background:rgba(126,206,202,.06); }\n.sidebar-status b { font-size:.72rem; color:#DDEDEA; }\n.sidebar-status span { font-size:.62rem; color:#77BDB9; }\n.pipeline-node { display:grid; grid-template-columns:30px 1fr; gap:.7rem; padding:.8rem .15rem; position:relative; }\n.pipeline-node:not(:last-child)::after { content:\'\'; position:absolute; left:14px; top:37px; width:1px; height:34px; background:linear-gradient(#4AA8A5,rgba(74,168,165,.08)); }\n.node-num { width:29px; height:29px; border-radius:9px; display:grid; place-items:center; font:600 .62rem \'IBM Plex Mono\',monospace; color:#A8E2DE; border:1px solid rgba(126,206,202,.26); background:rgba(126,206,202,.08); }\n.node-num.warm { color:#E9B96D; border-color:rgba(233,185,109,.24); background:rgba(233,185,109,.07); }\n.node-copy b { display:block; color:#F2F7FA; font-size:.77rem; line-height:1.25; }\n.node-copy small { display:block; color:#71869C; font-size:.65rem; margin-top:.2rem; line-height:1.35; }\n.range-list { display:grid; gap:.38rem; }\n.range-row { display:flex; justify-content:space-between; gap:.5rem; font-size:.68rem; padding:.45rem .05rem; border-bottom:1px solid rgba(255,255,255,.055); }\n.range-row span:first-child { color:#8193A7; } .range-row span:last-child { color:#D7E3EA; font-family:\'IBM Plex Mono\',monospace; font-size:.63rem; }\n.sidebar-foot { margin-top:1.7rem; padding-top:1rem; border-top:1px solid rgba(255,255,255,.07); color:#51677C; font-size:.62rem; line-height:1.6; }\n\n/* Top bar */\n.av-topbar { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.35rem .1rem 1rem; }\n.av-brand { display:flex; align-items:center; gap:.66rem; }\n.av-brand .av-logo-svg { width:38px; height:38px; }\n.av-brand-name { font:800 1.02rem/1 \'Manrope\',sans-serif; letter-spacing:-.035em; color:var(--navy-950); }\n.av-brand-name span { color:var(--teal-600); }\n.av-brand-tagline { font-size:.65rem; color:var(--faint); margin-top:.22rem; }\n.topbar-meta { display:flex; align-items:center; gap:.55rem; }\n.meta-pill { border:1px solid var(--line); border-radius:999px; padding:.42rem .7rem; background:rgba(255,255,255,.74); color:var(--muted); font-size:.65rem; font-weight:600; box-shadow:0 3px 12px rgba(28,46,74,.04); }\n.meta-pill.accent { color:var(--teal-700); border-color:rgba(46,125,123,.25); background:rgba(223,245,243,.66); }\n.av-topbar { position:sticky; top:.6rem; z-index:40; padding:.7rem 1.1rem; margin-bottom:1.15rem; background:rgba(255,255,255,.86); backdrop-filter:blur(14px); border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow-soft); }\n.navbar-links { display:flex; align-items:center; gap:1.6rem; }\n.nav-link { color:var(--muted); font-size:.78rem; font-weight:600; text-decoration:none; transition:color .18s ease; }\n.nav-link:hover { color:var(--teal-700); }\n.navbar-actions { display:flex; align-items:center; gap:.7rem; }\n.navbar-cta { display:inline-flex; align-items:center; gap:.4rem; padding:.55rem .9rem; border-radius:10px; background:linear-gradient(105deg,var(--navy-900),var(--teal-600)); color:#FFF; font-size:.74rem; font-weight:700; text-decoration:none; box-shadow:0 8px 20px rgba(37,107,105,.22); transition:transform .18s ease, box-shadow .18s ease; }\n.navbar-cta:hover { transform:translateY(-1px); box-shadow:0 12px 26px rgba(37,107,105,.28); color:#FFF; }\n@media (max-width:720px){.navbar-links{display:none}.av-topbar{top:0;border-radius:0;margin:0 -1rem 1rem;padding:.65rem 1rem}}\n.av-footer { margin:3.4rem -2.1rem 0; padding:2.5rem 2.6rem 1.5rem; color:#CBD9E2; background:linear-gradient(150deg,var(--navy-950),var(--navy-900) 60%,var(--navy-800)); }\n.footer-top { display:flex; justify-content:space-between; gap:2.5rem; flex-wrap:wrap; padding-bottom:1.6rem; border-bottom:1px solid rgba(255,255,255,.08); }\n.footer-brand { display:flex; align-items:center; gap:.7rem; max-width:340px; }\n.footer-brand .av-logo-svg { width:38px; height:38px; flex:none; }\n.footer-brand-name { font-size:.98rem; font-weight:800; line-height:1; color:#FFF; letter-spacing:-.03em; }\n.footer-brand-name span { color:#8ED7D2; }\n.footer-brand-sub { color:#8FA8B9; font-size:.68rem; margin-top:.32rem; line-height:1.5; }\n.footer-links { display:flex; gap:3.2rem; flex-wrap:wrap; }\n.footer-col { display:flex; flex-direction:column; gap:.5rem; }\n.footer-col-label { color:#8ED7D2; font-size:.62rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.15rem; }\n.footer-col a, .footer-col .footer-static { color:#B7C7D3; font-size:.73rem; text-decoration:none; }\n.footer-col a:hover { color:#FFF; }\n.footer-bottom { display:flex; justify-content:space-between; flex-wrap:wrap; gap:.5rem; padding-top:1.2rem; color:#6E869A; font-size:.65rem; }\n@media (max-width:720px){.av-footer{margin:2.4rem -1rem 0;padding:2rem 1.3rem 1.3rem}.footer-top{flex-direction:column;gap:1.4rem}.footer-links{gap:1.8rem}}\n\n/* Hero */\n.av-hero { position:relative; overflow:hidden; display:grid; grid-template-columns:minmax(0,1.15fr) minmax(350px,.85fr); gap:2.8rem; min-height:450px; padding:4rem 4.15rem; border-radius:30px; color:white; background:linear-gradient(130deg,#12243C 0%,#1C3755 54%,#245F62 116%); box-shadow:0 30px 78px rgba(20,39,64,.22); }\n.av-hero::before { content:\'\'; position:absolute; inset:0; background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px); background-size:38px 38px; mask-image:linear-gradient(to right,black,transparent 82%); }\n.av-hero::after { content:\'\'; position:absolute; width:430px; height:430px; right:-165px; top:-200px; border-radius:50%; background:radial-gradient(circle,rgba(126,206,202,.27),transparent 66%); }\n.hero-copy,.hero-visual { position:relative; z-index:2; }\n.hero-brand-lockup { display:flex; align-items:center; gap:.8rem; margin-bottom:1.15rem; }\n.hero-brand-lockup .av-logo-svg { width:56px; height:56px; flex:none; filter:drop-shadow(0 10px 22px rgba(3,15,30,.28)); }\n.hero-brand-name { color:#FFFFFF; font:800 1.38rem/1 \'Manrope\',sans-serif; letter-spacing:-.045em; }\n.hero-brand-name span { color:#91DCD7; }\n.hero-brand-sub { color:#8FA8B9; font-size:.65rem; margin-top:.28rem; letter-spacing:.025em; }\n.hero-eyebrow { display:inline-flex; align-items:center; gap:.45rem; color:#A6E0DC; font:700 .68rem \'Manrope\',sans-serif; letter-spacing:.14em; text-transform:uppercase; }\n.hero-eyebrow::before { content:\'\'; width:24px; height:1px; background:#7ECECA; }\n.av-hero h1 { max-width:680px; margin:1.15rem 0 1.05rem; color:#FFF; font:800 clamp(2.35rem,5vw,4.15rem)/1.01 \'Manrope\',sans-serif; letter-spacing:-.055em; }\n.av-hero h1 span { color:#91DCD7; }\n.hero-lead { max-width:665px; color:#C3D5E0; font-size:1.04rem; line-height:1.72; }\n.hero-chips { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.55rem; }\n.hero-chip { padding:.48rem .7rem; border-radius:999px; border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.055); color:#D5E2E9; font-size:.67rem; font-weight:600; backdrop-filter:blur(8px); }\n.hero-visual { display:grid; place-items:center; min-height:315px; }\n.hero-orbit { position:absolute; border:1px solid rgba(126,206,202,.13); border-radius:50%; }\n.orbit-one { width:350px; height:350px; } .orbit-two { width:245px; height:245px; }\n.hero-panel { width:min(100%,385px); padding:1rem; border:1px solid rgba(255,255,255,.13); border-radius:20px; background:linear-gradient(145deg,rgba(255,255,255,.115),rgba(255,255,255,.045)); box-shadow:0 24px 64px rgba(3,15,30,.30); backdrop-filter:blur(16px); }\n.hero-panel-head,.hero-panel-footer { display:flex; align-items:center; justify-content:space-between; color:#91A9BA; font-size:.61rem; font-weight:600; padding:.05rem .15rem .75rem; }\n.live-dot { color:#A9DDD9; display:flex; align-items:center; gap:.32rem; }.live-dot i{display:block;width:6px;height:6px;border-radius:50%;background:#7ECECA;box-shadow:0 0 0 4px rgba(126,206,202,.12)}\n.signal-row { display:grid; grid-template-columns:32px 1fr 72px; align-items:center; gap:.7rem; padding:.82rem .75rem; margin:.45rem 0; border-radius:13px; background:rgba(5,20,36,.27); border:1px solid rgba(255,255,255,.055); }\n.signal-icon { width:30px;height:30px;border-radius:9px;display:grid;place-items:center;background:rgba(126,206,202,.12);border:1px solid rgba(126,206,202,.18);color:#A7E0DC;font:600 .6rem \'IBM Plex Mono\',monospace; }.signal-icon.warm{color:#F0C27E;background:rgba(240,194,126,.10);border-color:rgba(240,194,126,.17)}\n.signal-copy b { display:block;color:#F0F6F8;font-size:.7rem; }.signal-copy small{display:block;color:#7890A2;font-size:.58rem;margin-top:.15rem;}\n.signal-wave { height:5px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden; }.signal-wave span{display:block;height:100%;border-radius:inherit;background:linear-gradient(90deg,#7ECECA,#4AA8A5)}.signal-wave.warm span{background:linear-gradient(90deg,#DDA95F,#F0C27E)}\n.hero-panel-footer { padding:.7rem .15rem .05rem; border-top:1px solid rgba(255,255,255,.07); margin-top:.65rem; }\n\n/* Trust strip */\n.trust-strip { display:grid; grid-template-columns:repeat(3,1fr); gap:1px; margin:1.1rem 0 2.4rem; overflow:hidden; border:1px solid var(--line); border-radius:17px; background:var(--line); box-shadow:var(--shadow-soft); }\n.trust-item { display:flex; align-items:center; gap:.72rem; background:rgba(255,255,255,.92); padding:1rem 1.15rem; }\n.trust-icon { width:34px;height:34px;border-radius:10px;display:grid;place-items:center;color:var(--teal-700);background:var(--aqua-100);font:700 .66rem \'IBM Plex Mono\',monospace; }\n.trust-icon svg { width:18px;height:18px;display:block; }\n.trust-copy b { display:block;font:700 .76rem \'Manrope\',sans-serif;color:var(--navy-900); }.trust-copy small{display:block;color:var(--faint);font-size:.66rem;margin-top:.15rem;}\n\n/* Sections */\n.section-head { display:flex; justify-content:space-between; gap:1rem; align-items:flex-end; margin:2.25rem 0 1rem; }\n.section-kicker { color:var(--teal-700); font:700 .65rem \'Manrope\',sans-serif; letter-spacing:.16em; text-transform:uppercase; margin-bottom:.42rem; }\n.section-head h2 { margin:0; color:var(--navy-950); font:800 clamp(1.45rem,3vw,2.05rem)/1.1 \'Manrope\',sans-serif; letter-spacing:-.04em; }\n.section-head p { max-width:520px; margin:.55rem 0 0; color:var(--muted); font-size:.83rem; line-height:1.55; }\n.section-tag { flex:none; border:1px solid var(--line); border-radius:999px; padding:.44rem .72rem; color:var(--muted); background:#FFF; font-size:.64rem; font-weight:600; }\n\n/* Streamlit form and widgets */\n[data-testid="stForm"] { background:rgba(255,255,255,.82); border:1px solid var(--line); border-radius:22px; padding:1.15rem 1.15rem .3rem; box-shadow:var(--shadow); backdrop-filter:blur(8px); }\ndiv[data-testid="stVerticalBlockBorderWrapper"] { background:#FFF; border:1px solid #DEE6EA !important; border-radius:17px !important; box-shadow:0 6px 20px rgba(28,46,74,.045); }\n.form-card-head { display:flex;align-items:center;gap:.72rem;padding:.2rem 0 .75rem;margin-bottom:.15rem;border-bottom:1px solid #EDF1F3; }\n.form-card-num { width:30px;height:30px;border-radius:9px;display:grid;place-items:center;background:var(--aqua-100);color:var(--teal-700);font:700 .62rem \'IBM Plex Mono\',monospace; }\n.form-card-title b { display:block;color:var(--navy-900);font:700 .78rem \'Manrope\',sans-serif; }.form-card-title small{display:block;color:var(--faint);font-size:.62rem;margin-top:.1rem;}\nlabel[data-testid="stWidgetLabel"] p { color:#526176 !important; font-size:.73rem !important; font-weight:600 !important; }\n[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input { background:#F6F8F9 !important; border:1px solid #DDE5E9 !important; border-radius:10px !important; color:var(--ink) !important; font-size:.78rem !important; min-height:42px; }\n[data-testid="stTextInput"] input:focus,[data-testid="stNumberInput"] input:focus { border-color:var(--teal-500) !important; box-shadow:0 0 0 3px rgba(74,168,165,.12) !important; }\n[data-testid="stSelectbox"] > div > div { background:#F6F8F9 !important; border:1px solid #DDE5E9 !important; border-radius:10px !important; min-height:42px; }\n[data-testid="stSelectbox"] div[data-baseweb="select"],\n[data-testid="stSelectbox"] div[data-baseweb="select"] > div,\n[data-testid="stSelectbox"] div[data-baseweb="select"] div,\n[data-testid="stSelectbox"] [data-baseweb="base-input"] { background:#F6F8F9 !important; background-color:#F6F8F9 !important; }\n/* Keep selected dropdown values readable on the light control background. */\n[data-testid="stSelectbox"] [data-baseweb="select"] div,\n[data-testid="stSelectbox"] [data-baseweb="select"] span { color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; opacity:1 !important; font-size:.77rem !important; }\n[data-testid="stSelectbox"] [data-baseweb="select"] svg { color:var(--muted) !important; fill:var(--muted) !important; opacity:1 !important; }\n/* Keep the opened dropdown menu readable as well. */\n[data-baseweb="popover"] { color:var(--ink) !important; }\n[data-baseweb="popover"] > div,\n[data-baseweb="popover"] [data-baseweb="menu"],\n[data-baseweb="popover"] ul,\n[data-baseweb="popover"] [role="listbox"] { background:#FFFFFF !important; background-color:#FFFFFF !important; }\n[data-baseweb="popover"] [role="option"],\n[data-baseweb="popover"] [role="option"] * { color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; opacity:1 !important; font-size:.77rem !important; }\n[data-baseweb="popover"] [role="option"]:hover,\n[data-baseweb="popover"] [aria-selected="true"] { background:var(--aqua-100) !important; }\n[data-testid="stFormSubmitButton"] > button,[data-testid="stButton"] > button { min-height:48px; border:0 !important; border-radius:12px !important; background:linear-gradient(105deg,var(--navy-900),var(--teal-600)) !important; color:#FFF !important; font:700 .8rem \'Manrope\',sans-serif !important; letter-spacing:.01em; box-shadow:0 10px 24px rgba(37,107,105,.22) !important; transition:.2s ease !important; }\n[data-testid="stFormSubmitButton"] > button:hover,[data-testid="stButton"] > button:hover { transform:translateY(-1px); box-shadow:0 14px 30px rgba(37,107,105,.29) !important; }\n.form-footnote { color:var(--faint);font-size:.63rem;line-height:1.45;padding:.2rem .1rem .75rem; }\n\n/* Process profile */\n.profile-grid { display:grid;grid-template-columns:repeat(4,1fr);gap:.7rem;margin-top:.9rem; }\n.profile-card { background:rgba(255,255,255,.84);border:1px solid var(--line);border-radius:14px;padding:.8rem .9rem;box-shadow:0 5px 16px rgba(28,46,74,.04); }\n.profile-label { color:var(--faint);font-size:.58rem;font-weight:700;letter-spacing:.11em;text-transform:uppercase; }\n.profile-value { color:var(--navy-900);font:700 .82rem \'Manrope\',sans-serif;margin-top:.34rem; }.profile-value.good{color:var(--safe)}.profile-value.watch{color:var(--warn)}\n\n/* Results */\n.result-grid { display:grid;grid-template-columns:repeat(3,1fr);gap:.85rem; }\n.result-card { position:relative;overflow:hidden;background:#FFF;border:1px solid var(--line);border-radius:18px;padding:1.35rem 1.25rem;box-shadow:var(--shadow-soft);min-height:180px; }\n.result-card::before { content:\'\';position:absolute;inset:0 0 auto;height:4px;background:linear-gradient(90deg,var(--teal-500),var(--teal-700)); }.result-card.safe::before{background:linear-gradient(90deg,#43A17F,#26705A)}.result-card.warn::before{background:linear-gradient(90deg,#D39A4D,#A96B1C)}.result-card.danger::before{background:linear-gradient(90deg,#C75A5A,#963B3B)}.result-card.neutral::before{background:linear-gradient(90deg,#9AA8B6,#657489)}\n.result-top { display:flex;justify-content:space-between;align-items:center;gap:.5rem; }.result-index{font:600 .62rem \'IBM Plex Mono\',monospace;color:var(--faint)}\n.status-badge { padding:.3rem .52rem;border-radius:999px;font-size:.58rem;font-weight:700;background:var(--aqua-100);color:var(--teal-700); }.status-badge.safe{background:#E6F4EF;color:var(--safe)}.status-badge.warn{background:#FFF3E3;color:var(--warn)}.status-badge.danger{background:#FBEAEA;color:var(--danger)}.status-badge.neutral{background:#EEF2F5;color:var(--muted)}\n.result-label { color:var(--muted);font-size:.68rem;font-weight:700;margin-top:1.2rem; }\n.result-value { color:var(--navy-950);font:700 clamp(1.75rem,3vw,2.65rem)/1 \'IBM Plex Mono\',monospace;letter-spacing:-.05em;margin-top:.55rem; }.result-value.safe{color:var(--safe)}.result-value.warn{color:var(--warn)}.result-value.danger{color:var(--danger)}.result-value.teal{color:var(--teal-700)}\n.result-unit { color:var(--faint);font-size:.65rem;margin-top:.48rem; }\n.insight-card { display:grid;grid-template-columns:38px 1fr;gap:.85rem;margin-top:.9rem;padding:1.1rem 1.2rem;border:1px solid var(--line);border-left:4px solid var(--teal-600);border-radius:15px;background:#FFF;box-shadow:var(--shadow-soft); }.insight-card.safe{border-left-color:var(--safe)}.insight-card.warn{border-left-color:var(--warn)}.insight-card.danger{border-left-color:var(--danger)}.insight-card.neutral{border-left-color:var(--muted)}\n.insight-icon { width:36px;height:36px;border-radius:10px;display:grid;place-items:center;background:var(--aqua-100);color:var(--teal-700);font:700 .65rem \'IBM Plex Mono\',monospace; }\n.insight-icon svg { width:20px;height:20px;display:block; }\n.insight-label { color:var(--faint);font-size:.59rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase; }.insight-copy{color:var(--ink);font-size:.84rem;line-height:1.58;margin-top:.3rem;}\n.probability-panel { background:#FFF;border:1px solid var(--line);border-radius:17px;padding:1rem 1.15rem;box-shadow:var(--shadow-soft); }\n.prob-row { display:grid;grid-template-columns:145px 1fr 48px;align-items:center;gap:.8rem;padding:.62rem 0;border-bottom:1px solid #EEF2F4; }.prob-row:last-child{border-bottom:0}.prob-name{font-size:.69rem;color:var(--muted);font-weight:600}.prob-name.top{color:var(--navy-900)}.prob-track{height:7px;border-radius:999px;background:#EDF2F3;overflow:hidden}.prob-fill{height:100%;border-radius:inherit;background:linear-gradient(90deg,var(--aqua-300),var(--teal-600))}.prob-fill.top{background:linear-gradient(90deg,var(--teal-500),var(--safe))}.prob-pct{text-align:right;color:var(--muted);font:600 .64rem \'IBM Plex Mono\',monospace}\n.empty-state { padding:3.2rem 1.5rem;text-align:center;border:1px dashed #CBD7DD;border-radius:20px;background:rgba(255,255,255,.58); }.empty-icon{width:50px;height:50px;margin:0 auto .85rem;border-radius:15px;display:grid;place-items:center;background:var(--aqua-100);color:var(--teal-700);font:700 .7rem \'IBM Plex Mono\',monospace}\n.empty-icon svg { width:26px;height:26px;display:block; }.empty-state b{display:block;color:var(--navy-900);font:700 1rem \'Manrope\',sans-serif}.empty-state p{max-width:460px;margin:.45rem auto 0;color:var(--muted);font-size:.77rem;line-height:1.55}\n[data-testid="stExpander"] { background:#FFF !important;border:1px solid var(--line) !important;border-radius:14px !important;box-shadow:0 5px 16px rgba(28,46,74,.04) !important; }\n.input-summary { display:grid;grid-template-columns:repeat(2,1fr);gap:.55rem 1.5rem;color:var(--muted);font-size:.72rem;line-height:1.7; }.input-summary b{color:var(--navy-900)}\n\n@media (max-width:980px){.av-hero{grid-template-columns:1fr;padding:2.5rem;}.hero-visual{min-height:240px}.trust-strip{grid-template-columns:1fr}.result-grid{grid-template-columns:1fr}.profile-grid{grid-template-columns:repeat(2,1fr)}}\n@media (max-width:720px){.block-container{padding:.7rem 1rem 3rem !important}.av-topbar{align-items:flex-start}.topbar-meta{display:none}.av-hero{padding:2rem 1.35rem;border-radius:21px}.av-hero h1{font-size:2.35rem}.hero-panel{width:100%}.section-head{align-items:flex-start;flex-direction:column}.profile-grid{grid-template-columns:1fr 1fr}.prob-row{grid-template-columns:100px 1fr 42px}.input-summary{grid-template-columns:1fr}}\n\n/* Bioprocess visual language */\n.bio-visual { position:relative; z-index:2; display:grid; place-items:center; min-height:300px; }\n.bio-orbit { position:absolute; border:1px solid rgba(126,206,202,.12); border-radius:50%; }\n.bio-orbit-one { width:430px; height:430px; }\n.bio-orbit-two { width:305px; height:305px; }\n.bio-panel { position:relative; width:min(100%,505px); padding:1rem 1rem .85rem; border:1px solid rgba(255,255,255,.13); border-radius:22px; background:linear-gradient(145deg,rgba(255,255,255,.11),rgba(255,255,255,.04)); box-shadow:0 26px 70px rgba(3,15,30,.31); backdrop-filter:blur(18px); }\n.bio-panel-head,.bio-panel-footer { display:flex; align-items:center; justify-content:space-between; gap:1rem; color:#91A9BA; font-size:.59rem; font-weight:600; padding:.05rem .15rem .75rem; }\n.bio-process-map { display:grid; grid-template-columns:minmax(108px,.8fr) minmax(122px,1fr) minmax(118px,.9fr); align-items:center; gap:.55rem; padding:.35rem .2rem .55rem; }\n.bio-inputs,.bio-outputs { display:grid; gap:.58rem; }\n.bio-node { position:relative; padding:.65rem .65rem; border-radius:12px; border:1px solid rgba(255,255,255,.07); background:rgba(4,20,36,.26); }\n.bio-node b { display:block; color:#ECF6F7; font-size:.64rem; line-height:1.25; }\n.bio-node small { display:block; color:#7890A2; font-size:.52rem; line-height:1.35; margin-top:.18rem; }\n.bio-node-code { display:block; margin-bottom:.32rem; color:#8FD8D3; font:600 .48rem \'IBM Plex Mono\',monospace; letter-spacing:.08em; }\n.bio-inputs .bio-node::after { content:\'\'; position:absolute; top:50%; right:-13px; width:12px; height:1px; background:linear-gradient(90deg,#7ECECA,rgba(126,206,202,.12)); }\n.bio-outputs .bio-node::before { content:\'\'; position:absolute; top:50%; left:-13px; width:12px; height:1px; background:linear-gradient(90deg,rgba(126,206,202,.12),#7ECECA); }\n.bio-node.output { display:grid; grid-template-columns:7px 1fr; align-items:center; gap:.5rem; padding:.62rem .6rem; }\n.bio-dot { width:7px; height:7px; border-radius:50%; background:#7ECECA; box-shadow:0 0 0 4px rgba(126,206,202,.09); }\n.bio-node.output.warm .bio-dot { background:#E8B96F; box-shadow:0 0 0 4px rgba(232,185,111,.09); }\n.reactor-wrap { display:grid; place-items:center; min-width:0; }\n.reactor-svg { width:100%; max-width:155px; height:auto; filter:drop-shadow(0 14px 22px rgba(3,15,30,.22)); }\n.reactor-caption { text-align:center; margin-top:-.25rem; }\n.reactor-caption b { display:block; color:#EAF5F6; font-size:.61rem; }\n.reactor-caption span { display:block; color:#7890A2; font-size:.5rem; margin-top:.12rem; }\n.bio-panel-footer { padding:.72rem .15rem .02rem; margin-top:.25rem; border-top:1px solid rgba(255,255,255,.07); }\n\n.optimization-loop { display:grid; grid-template-columns:170px repeat(4,1fr); align-items:stretch; margin:1.1rem 0 3.4rem; border:1px solid var(--line); border-radius:19px; background:rgba(255,255,255,.82); box-shadow:var(--shadow-soft); overflow:hidden; }\n.loop-intro { padding:1rem 1.05rem; background:linear-gradient(145deg,var(--navy-950),var(--navy-800)); }\n.loop-intro span { display:block; color:#8ED7D2; font:700 .55rem \'Manrope\',sans-serif; letter-spacing:.12em; text-transform:uppercase; }\n.loop-intro b { display:block; color:#FFF; font:700 .82rem/1.3 \'Manrope\',sans-serif; margin-top:.42rem; }\n.loop-step { position:relative; padding:1rem .95rem; border-left:1px solid #E3E9EC; }\n.loop-step:not(:last-child)::after { content:\'→\'; position:absolute; right:-.35rem; top:50%; transform:translateY(-50%); z-index:2; width:.72rem; height:1.15rem; display:grid; place-items:center; color:var(--teal-500); background:#FFF; font-size:.65rem; }\n.loop-num { color:var(--teal-600); font:600 .56rem \'IBM Plex Mono\',monospace; }\n.loop-step b { display:block; color:var(--navy-900); font:700 .69rem/1.25 \'Manrope\',sans-serif; margin-top:.38rem; }\n.loop-step small { display:block; color:var(--muted); font-size:.58rem; line-height:1.38; margin-top:.2rem; }\n\n.bioprocess-ribbon { display:flex; align-items:center; flex-wrap:wrap; gap:.48rem; margin:.78rem 0 .25rem; }\n.bio-ribbon-label { color:var(--faint); font-size:.58rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin-right:.15rem; }\n.bio-ribbon-pill { display:inline-flex; align-items:center; gap:.35rem; padding:.34rem .52rem; border:1px solid var(--line); border-radius:999px; background:#FFF; color:var(--muted); font-size:.59rem; font-weight:600; }\n.bio-ribbon-pill i { width:5px; height:5px; border-radius:50%; background:var(--teal-500); }\n\n@media (max-width:1100px){.optimization-loop{grid-template-columns:150px repeat(2,1fr)}.loop-step:nth-child(4){border-top:1px solid #E3E9EC}.loop-step:nth-child(5){border-top:1px solid #E3E9EC}.bio-process-map{grid-template-columns:1fr 1.1fr}.bio-outputs{grid-column:1/-1;grid-template-columns:repeat(3,1fr)}.bio-outputs .bio-node::before{display:none}.bio-inputs .bio-node::after{right:-10px;width:9px}}\n@media (max-width:720px){.bio-process-map{grid-template-columns:1fr}.bio-inputs,.bio-outputs{grid-template-columns:1fr 1fr}.bio-outputs{grid-template-columns:1fr}.bio-inputs .bio-node::after{display:none}.optimization-loop{grid-template-columns:1fr 1fr}.loop-intro{grid-column:1/-1}.loop-step:nth-child(n){border-top:1px solid #E3E9EC}.loop-step:not(:last-child)::after{display:none}}\n\n/* ============================================================\n   Micro-interactions, hover feedback & ambient motion\n   ============================================================ */\n\n@keyframes avFadeUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }\n@keyframes avOrbitSpin { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }\n@keyframes avPulseDot { 0%,100% { box-shadow:0 0 0 0 rgba(126,206,202,.35); } 50% { box-shadow:0 0 0 6px rgba(126,206,202,0); } }\n@keyframes avBubbleRise { 0% { transform:translateY(0) scale(.85); opacity:.35; } 45% { opacity:1; } 100% { transform:translateY(-14px) scale(1.05); opacity:0; } }\n@keyframes avGlowPulse { 0%,100% { opacity:.75; transform:scale(1); } 50% { opacity:1; transform:scale(1.12); } }\n@keyframes avShimmer { 0% { transform:translateX(-120%); } 100% { transform:translateX(220%); } }\n@keyframes avHeadlineGlow { 0%,100% { filter:brightness(1); } 50% { filter:brightness(1.22); } }\n\n.av-hero, .trust-strip, .result-card, .insight-card, .empty-state { animation:avFadeUp .55s cubic-bezier(.2,.7,.3,1) both; }\n.trust-item:nth-child(1){animation:avFadeUp .5s .04s cubic-bezier(.2,.7,.3,1) both;}\n.trust-item:nth-child(2){animation:avFadeUp .5s .1s cubic-bezier(.2,.7,.3,1) both;}\n.trust-item:nth-child(3){animation:avFadeUp .5s .16s cubic-bezier(.2,.7,.3,1) both;}\n.result-card:nth-child(1){animation-delay:.02s;} .result-card:nth-child(2){animation-delay:.08s;} .result-card:nth-child(3){animation-delay:.14s;}\n\n.bio-orbit-one, .bio-orbit-two { animation:avOrbitSpin 90s linear infinite; }\n.bio-orbit-two { animation-duration:70s; animation-direction:reverse; }\n.live-dot i { animation:avPulseDot 2.1s ease-out infinite; }\n.bio-dot { animation:avPulseDot 2.4s ease-out infinite; }\n.bio-bubble { animation-name:avBubbleRise; animation-duration:3.2s; animation-timing-function:ease-in-out; animation-iteration-count:infinite; transform-box:fill-box; transform-origin:center; }\n.bio-glow-dot { animation-name:avGlowPulse; animation-duration:2.6s; animation-timing-function:ease-in-out; animation-iteration-count:infinite; transform-box:fill-box; transform-origin:center; }\n.av-hero h1 span { background:linear-gradient(90deg,#91DCD7,#C9F3EF,#91DCD7); background-size:220% auto; -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; animation:avHeadlineGlow 5s ease-in-out infinite; }\n\n.trust-item, .profile-card, .result-card, .insight-card, .loop-step, .bio-ribbon-pill, .meta-pill, .hero-chip, .pipeline-node, .range-row, .sidebar-status, .prob-row, .bio-node {\n  transition:transform .22s cubic-bezier(.2,.7,.3,1), box-shadow .22s ease, border-color .22s ease, background-color .22s ease;\n}\n.trust-item:hover { transform:translateY(-3px); box-shadow:0 14px 28px rgba(28,46,74,.09); z-index:1; position:relative; }\n.trust-item:hover .trust-icon { background:var(--teal-600); color:#FFF; transform:rotate(-6deg); transition:transform .22s ease, background-color .22s ease, color .22s ease; }\n.profile-card:hover { transform:translateY(-3px); box-shadow:0 12px 26px rgba(28,46,74,.09); border-color:rgba(74,168,165,.4); }\n.result-card { cursor:default; }\n.result-card:hover { transform:translateY(-4px); box-shadow:0 20px 40px rgba(28,46,74,.13); border-color:rgba(74,168,165,.35); }\n.result-card:hover .result-value { transform:scale(1.03); transition:transform .22s ease; }\n.insight-card:hover { transform:translateY(-2px); box-shadow:0 16px 32px rgba(28,46,74,.1); }\n.loop-step:hover { background:rgba(126,206,202,.06); }\n.loop-step:hover .loop-num { color:var(--teal-700); }\n.bio-ribbon-pill:hover, .meta-pill:hover { transform:translateY(-2px); box-shadow:0 8px 18px rgba(28,46,74,.08); border-color:rgba(74,168,165,.4); }\n.hero-chip:hover { transform:translateY(-2px); background:rgba(255,255,255,.12); border-color:rgba(126,206,202,.4); }\n.pipeline-node:hover, .range-row:hover { background:rgba(126,206,202,.06); border-radius:8px; }\n.sidebar-status:hover { border-color:rgba(126,206,202,.32); background:rgba(126,206,202,.1); }\n.prob-row:hover { background:rgba(126,206,202,.05); border-radius:8px; }\n.prob-fill { position:relative; overflow:hidden; transition:width .7s cubic-bezier(.2,.7,.3,1); }\n.prob-fill::after { content:\'\'; position:absolute; inset:0; width:40%; background:linear-gradient(90deg,transparent,rgba(255,255,255,.55),transparent); animation:avShimmer 2.4s ease-in-out infinite; }\n.bio-node:hover { border-color:rgba(126,206,202,.28); background:rgba(126,206,202,.08); }\n.reactor-svg { transition:transform .3s ease; }\n.bio-panel:hover .reactor-svg { transform:scale(1.03); }\n\na:focus-visible, button:focus-visible, [tabindex]:focus-visible { outline:2px solid var(--teal-500) !important; outline-offset:2px; }\n\n@media (prefers-reduced-motion: reduce) {\n  .av-hero, .trust-strip, .result-card, .insight-card, .empty-state, .trust-item, .bio-orbit-one, .bio-orbit-two, .live-dot i, .bio-dot, .bio-bubble, .bio-glow-dot, .av-hero h1 span, .prob-fill::after {\n    animation:none !important;\n  }\n  * { transition-duration:.01ms !important; }\n}\n\n</style>'
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TASK1_DIR = BASE_DIR / "final_task1_methane_yield_model"
TASK2_DIR = BASE_DIR / "final_task2_inhibition_risk_model"
TASK3_DIR = BASE_DIR / "final_task3_inhibition_type_model"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def load_feature_names(path):
    data = load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["feature_names", "features", "final_features", "columns"]:
            if key in data:
                return data[key]
    raise ValueError(f"Could not read feature names from: {path}")

def apply_target_encoder(encoder, substrate_value):
    substrate_df = pd.DataFrame({"Substrate": [substrate_value]})
    encoded = encoder.transform(substrate_df)
    if isinstance(encoded, pd.DataFrame):
        return float(encoded.iloc[0, 0])
    return float(encoded[0][0])


# ============================================================
# LOAD MODELS (cached)
# ============================================================

@st.cache_resource
def load_all_models():
    task1_model   = joblib.load(TASK1_DIR / "ann_tuned_task1_model.pkl")
    task1_imputer = joblib.load(TASK1_DIR / "median_imputer_task1.pkl")
    task1_scaler  = joblib.load(TASK1_DIR / "standard_scaler_task1.pkl")
    task1_encoder = joblib.load(TASK1_DIR / "target_encoder_task1.pkl")
    task1_reactor_mapping = load_json(TASK1_DIR / "reactor_mapping.json")
    task1_feature_names   = load_feature_names(TASK1_DIR / "task1_feature_names.json")

    task2_model   = joblib.load(TASK2_DIR / "ann_mlp_task2_final_model.pkl")
    task2_imputer = joblib.load(TASK2_DIR / "median_imputer_task2_final.pkl")
    task2_scaler  = joblib.load(TASK2_DIR / "standard_scaler_task2_final.pkl")
    task2_encoder = joblib.load(TASK2_DIR / "target_encoder_task2_final.pkl")
    task2_reactor_mapping = load_json(TASK2_DIR / "reactor_mapping_task2_final.json")
    task2_feature_names   = load_feature_names(TASK2_DIR / "task2_final_feature_names.json")

    task2_threshold = 0.14
    task2_summary_path = TASK2_DIR / "task2_final_model_summary.json"
    if task2_summary_path.exists():
        task2_summary = load_json(task2_summary_path)
        for key in ["threshold", "decision_threshold", "final_threshold", "selected_threshold"]:
            if key in task2_summary:
                task2_threshold = float(task2_summary[key])
                break

    task3_model = joblib.load(TASK3_DIR / "task3_final_tuned_rf_pipeline.pkl")
    task3_label_mapping_raw = load_json(TASK3_DIR / "task3_label_mapping.json")
    task3_label_mapping = {int(k): v for k, v in task3_label_mapping_raw.items()}
    task3_thresholds = load_json(TASK3_DIR / "task3_confidence_thresholds.json")

    return {
        "task1_model": task1_model, "task1_imputer": task1_imputer,
        "task1_scaler": task1_scaler, "task1_encoder": task1_encoder,
        "task1_reactor_mapping": task1_reactor_mapping,
        "task1_feature_names": task1_feature_names,

        "task2_model": task2_model, "task2_imputer": task2_imputer,
        "task2_scaler": task2_scaler, "task2_encoder": task2_encoder,
        "task2_reactor_mapping": task2_reactor_mapping,
        "task2_feature_names": task2_feature_names,
        "task2_threshold": task2_threshold,

        "task3_model": task3_model,
        "task3_label_mapping": task3_label_mapping,
        "task3_thresholds": task3_thresholds,
    }

models = load_all_models()


# ============================================================
# PREDICTION FUNCTIONS  (logic unchanged)
# ============================================================

def predict_methane_yield(input_data):
    substrate_encoded = apply_target_encoder(models["task1_encoder"], input_data["Substrate"])
    impute_row = pd.DataFrame([{
        "HRT_days": input_data["HRT_days"],
        "OLR_gVS_L_day": input_data["OLR_gVS_L_day"],
        "ISR": input_data["ISR"],
    }])
    imputed = models["task1_imputer"].transform(impute_row)
    hrt_imp, olr_imp, isr_imp = imputed[0]
    scale_row = pd.DataFrame([{
        "Temperature_C": input_data["Temperature_C"],
        "pH": input_data["pH"],
        "HRT_days": hrt_imp,
        "OLR_gVS_L_day": olr_imp,
        "ISR": isr_imp,
        "TAN_mg_L": input_data["TAN_mg_L"],
        "VFA_mg_L": input_data["VFA_mg_L"],
        "Substrate": substrate_encoded,
    }])
    X_scaled = models["task1_scaler"].transform(scale_row)
    X_scaled_df = pd.DataFrame(X_scaled, columns=scale_row.columns)
    raw_reactor = input_data["Reactor_Type"]
    cleaned_reactor = models["task1_reactor_mapping"].get(raw_reactor, raw_reactor)
    for feature in models["task1_feature_names"]:
        if feature.startswith("Reactor_Type_"):
            X_scaled_df[feature] = 0
    reactor_feature = f"Reactor_Type_{cleaned_reactor}"
    if reactor_feature in models["task1_feature_names"]:
        X_scaled_df[reactor_feature] = 1
    X_final = X_scaled_df[models["task1_feature_names"]]
    prediction = models["task1_model"].predict(X_final)
    return float(np.ravel(prediction)[0])


def predict_inhibition_risk(input_data):
    substrate_encoded = apply_target_encoder(models["task2_encoder"], input_data["Substrate"])
    impute_row = pd.DataFrame([{
        "HRT_days": input_data["HRT_days"],
        "OLR_gVS_L_day": input_data["OLR_gVS_L_day"],
        "ISR": input_data["ISR"],
    }])
    imputed = models["task2_imputer"].transform(impute_row)
    hrt_imp, olr_imp, isr_imp = imputed[0]
    scale_row = pd.DataFrame([{
        "Temperature_C": input_data["Temperature_C"],
        "pH": input_data["pH"],
        "HRT_days": hrt_imp,
        "OLR_gVS_L_day": olr_imp,
        "ISR": isr_imp,
        "TAN_mg_L": input_data["TAN_mg_L"],
        "VFA_mg_L": input_data["VFA_mg_L"],
        "Substrate": substrate_encoded,
    }])
    X_scaled = models["task2_scaler"].transform(scale_row)
    X_scaled_df = pd.DataFrame(X_scaled, columns=scale_row.columns)
    raw_reactor = input_data["Reactor_Type"]
    cleaned_reactor = models["task2_reactor_mapping"].get(raw_reactor, raw_reactor)
    for feature in models["task2_feature_names"]:
        if feature.startswith("Reactor_Type_"):
            X_scaled_df[feature] = 0
    reactor_feature = f"Reactor_Type_{cleaned_reactor}"
    if reactor_feature in models["task2_feature_names"]:
        X_scaled_df[reactor_feature] = 1
    X_final = X_scaled_df[models["task2_feature_names"]]
    if hasattr(models["task2_model"], "predict_proba"):
        probability = float(models["task2_model"].predict_proba(X_final)[0][1])
    else:
        raw_prediction = models["task2_model"].predict(X_final)
        probability = float(np.ravel(raw_prediction)[0])
    threshold = models["task2_threshold"]
    risk_detected = probability >= threshold
    return probability, threshold, risk_detected


def clean_task3_reactor_type(reactor_type):
    reactor_mapping = {
        "Batch": "Batch", "Batch (BMP)": "Batch", "Batch serum bottles": "Batch",
        "Thermophilic Batch": "Thermophilic_Batch", "Thermophilic AD": "Thermophilic_Batch",
        "Thermophilic_Batch": "Thermophilic_Batch",
        "CSTR": "CSTR", "CSTR (fed-batch)": "CSTR", "CSTR (mesophilic)": "CSTR",
        "Two-stage CSTR": "CSTR", "Serial CSTR": "CSTR",
        "Horizontal flow reactor": "Other", "Horizontal flow": "Other", "Other": "Other"
    }
    return reactor_mapping.get(reactor_type, "Other")


def predict_inhibition_type(input_data):
    task3_input = {
        "Temperature_C": input_data["Temperature_C"],
        "pH": input_data["pH"],
        "HRT_days": input_data["HRT_days"],
        "OLR_gVS_L_day": input_data["OLR_gVS_L_day"],
        "ISR": input_data["ISR"],
        "TAN_mg_L": input_data["TAN_mg_L"],
        "VFA_mg_L": input_data["VFA_mg_L"],
        "Substrate_Category": input_data["Substrate_Category"],
        "Reactor_Type": clean_task3_reactor_type(input_data["Reactor_Type"])
    }
    X = pd.DataFrame([task3_input])
    predicted_encoded = int(models["task3_model"].predict(X)[0])
    predicted_type = models["task3_label_mapping"][predicted_encoded]
    probabilities = models["task3_model"].predict_proba(X)[0]
    confidence = float(probabilities[predicted_encoded])
    class_probabilities = {
        models["task3_label_mapping"][int(i)]: float(probabilities[i])
        for i in range(len(probabilities))
    }
    high_threshold     = float(models["task3_thresholds"].get("high_confidence_threshold", 0.60))
    moderate_threshold = float(models["task3_thresholds"].get("moderate_confidence_threshold", 0.45))
    low_threshold      = float(models["task3_thresholds"].get("low_confidence_threshold", 0.40))

    if confidence >= high_threshold:
        confidence_level = "High"
        final_message = f"Likely inhibition type: **{predicted_type}**"
    elif confidence >= moderate_threshold:
        confidence_level = "Moderate"
        final_message = f"Possible inhibition type: **{predicted_type}** — verify with process indicators."
    elif confidence >= low_threshold:
        confidence_level = "Low"
        final_message = f"Tentative inhibition type: **{predicted_type}** — interpret with caution."
    else:
        confidence_level = "Uncertain"
        final_message = "Inhibition type uncertain — additional process indicators required."

    return {
        "predicted_type": predicted_type,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "final_message": final_message,
        "class_probabilities": class_probabilities
    }


# ============================================================
# PRESENTATION HELPERS
# ============================================================

def process_profile(input_data):
    ph_state = ("Optimal", "good") if 6.8 <= input_data["pH"] <= 7.4 else ("Review", "watch")
    temp = input_data["Temperature_C"]
    if 35 <= temp <= 40:
        temp_state = ("Mesophilic", "good")
    elif 50 <= temp <= 60:
        temp_state = ("Thermophilic", "good")
    else:
        temp_state = ("Outside common band", "watch")
    tan_state = ("Elevated", "watch") if input_data["TAN_mg_L"] > 1500 else ("Within reference", "good")
    vfa_state = ("Elevated", "watch") if input_data["VFA_mg_L"] > 4000 else ("Within reference", "good")
    return ph_state, temp_state, tan_state, vfa_state


def render_section(kicker, title, description, tag=None):
    tag_html = f'<div class="section-tag">{html.escape(tag)}</div>' if tag else ""
    st.markdown(
        f'''<div class="section-head"><div><div class="section-kicker">{html.escape(kicker)}</div><h2>{html.escape(title)}</h2><p>{html.escape(description)}</p></div>{tag_html}</div>''',
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        f'''<div class="sidebar-brand">{LOGO_SVG}<div><div class="sidebar-wordmark">Anaero<span>Vision</span></div><div class="sidebar-sub">Bioprocess optimization intelligence</div></div></div>
        <div class="sidebar-status"><b>Prediction engine</b><span>● Models loaded</span></div>''',
        unsafe_allow_html=True,
    )
    st.markdown("### Model cascade")
    st.markdown(
        '''<div class="pipeline-node"><div class="node-num">01</div><div class="node-copy"><b>Methane yield</b><small>Tuned ANN regression</small></div></div>
        <div class="pipeline-node"><div class="node-num">02</div><div class="node-copy"><b>Inhibition screening</b><small>ANN binary classifier · threshold 0.14</small></div></div>
        <div class="pipeline-node"><div class="node-num warm">03</div><div class="node-copy"><b>Inhibition diagnosis</b><small>Tuned random forest · risk-triggered</small></div></div>''',
        unsafe_allow_html=True,
    )
    st.markdown("### Reference window")
    st.markdown(
        '''<div class="range-list">
        <div class="range-row"><span>Mesophilic</span><span>35–40 °C</span></div>
        <div class="range-row"><span>Thermophilic</span><span>50–60 °C</span></div>
        <div class="range-row"><span>Optimal pH</span><span>6.8–7.4</span></div>
        <div class="range-row"><span>TAN watch</span><span>&gt;1500 mg/L</span></div>
        <div class="range-row"><span>VFA watch</span><span>&gt;4000 mg/L</span></div>
        <div class="range-row"><span>Typical OLR</span><span>1–5 gVS/L/d</span></div>
        </div><div class="sidebar-foot">AnaeroVision v1.2<br>Bioprocess decision support for anaerobic digestion development.</div>''',
        unsafe_allow_html=True,
    )


# ============================================================
# BRAND HEADER + HERO
# ============================================================

st.markdown(
    f'''<div class="av-topbar">
    <nav class="navbar-links">
        <a class="nav-link" href="#how-it-works">How it works</a>
        <a class="nav-link" href="#predict">Predict</a>
        <a class="nav-link" href="#results">Results</a>
    </nav>
    <div class="navbar-actions">
        <span class="meta-pill accent">3-model cascade</span>
        <a class="navbar-cta" href="#predict">Run analysis →</a>
    </div>
    </div>''',
    unsafe_allow_html=True,
)

st.markdown(
    f'''<section class="av-hero"><div class="hero-copy"><div class="hero-brand-lockup">{LOGO_SVG}<div><div class="hero-brand-name">Anaero<span>Vision</span></div><div class="hero-brand-sub">AI-powered bioprocess optimization platform</div></div></div><div class="hero-eyebrow">Bioprocess intelligence for anaerobic digestion</div><h1>Optimize biology. <span>Engineer performance.</span></h1><div class="hero-lead">Estimate methane yield, screen for inhibition, and diagnose the most likely inhibition pathway through one connected prediction workflow.</div><div class="hero-chips"><span class="hero-chip">Bioconversion performance</span><span class="hero-chip">Process stability</span><span class="hero-chip">Root-cause diagnostics</span></div></div>{HERO_VISUAL}</section>''',
    unsafe_allow_html=True,
)

st.markdown(
    f'''<div class="trust-strip">
    <div class="trust-item"><div class="trust-icon">{ICON_WORKFLOW}</div><div class="trust-copy"><b>Biology-to-operation workflow</b><small>Link feedstock, reactor state, and biological response.</small></div></div>
    <div class="trust-item"><div class="trust-icon">{ICON_CHIP}</div><div class="trust-copy"><b>Engineering-grade visibility</b><small>Performance and uncertainty stay visible together.</small></div></div>
    <div class="trust-item"><div class="trust-icon">{ICON_FLASK}</div><div class="trust-copy"><b>Optimization mindset</b><small>The interface follows the way bioprocesses are engineered.</small></div></div>
    </div>''',
    unsafe_allow_html=True,
)

st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)
st.markdown(
    '''<div class="optimization-loop">
    <div class="loop-intro"><span>Closed-loop thinking</span><b>Bioprocess optimization pathway</b></div>
    <div class="loop-step"><div class="loop-num">01</div><b>Characterize feedstock</b><small>Translate substrate properties into a usable process context.</small></div>
    <div class="loop-step"><div class="loop-num">02</div><b>Set operating window</b><small>Define temperature, pH, retention time, and loading.</small></div>
    <div class="loop-step"><div class="loop-num">03</div><b>Predict biological response</b><small>Estimate conversion performance and stability pressure.</small></div>
    <div class="loop-step"><div class="loop-num">04</div><b>Support process action</b><small>Use confidence-aware outputs to guide the next engineering decision.</small></div>
    </div>''',
    unsafe_allow_html=True,
)


# ============================================================
# PREDICTION WORKSPACE
# ============================================================

st.markdown('<div id="predict"></div>', unsafe_allow_html=True)
render_section(
    "Bioprocess configuration",
    "Build the reactor state",
    "Characterize the feedstock, define the reactor operating window, and enter biological stability indicators. The full optimization cascade runs from one submission.",
    "9 process variables",
)

st.markdown(
    '''<div class="bioprocess-ribbon"><span class="bio-ribbon-label">Optimization lens</span><span class="bio-ribbon-pill"><i></i>Feedstock quality</span><span class="bio-ribbon-pill"><i></i>Hydraulic control</span><span class="bio-ribbon-pill"><i></i>Organic loading</span><span class="bio-ribbon-pill"><i></i>Microbial stability</span><span class="bio-ribbon-pill"><i></i>Gas productivity</span></div>''',
    unsafe_allow_html=True,
)

with st.form("anaerovision_prediction_form", clear_on_submit=False):
    col1, col2, col3 = st.columns([1.06, 1, 1])

    with col1:
        with st.container(border=True):
            st.markdown('<div class="form-card-head"><div class="form-card-num">01</div><div class="form-card-title"><b>Feedstock characterization</b><small>Describe the biological feed and vessel</small></div></div>', unsafe_allow_html=True)
            substrate = st.text_input("Substrate name", value="Protein-rich waste", help="Enter the substrate used in the AD process")
            substrate_category = st.selectbox("Substrate category", [
                "Protein-Rich", "Lipid-Rich", "Food Waste", "Manure/Sludge",
                "Agricultural residue", "Industrial organic waste", "Co-Digestion", "Other/unknown"
            ])
            reactor_type = st.selectbox("Reactor type", ["Batch", "CSTR", "Thermophilic_Batch", "Other"])

    with col2:
        with st.container(border=True):
            st.markdown('<div class="form-card-head"><div class="form-card-num">02</div><div class="form-card-title"><b>Reactor operating window</b><small>Define the engineered growth environment</small></div></div>', unsafe_allow_html=True)
            temperature = st.number_input("Temperature (°C)", min_value=10.0, max_value=70.0, value=37.0, step=0.5)
            ph = st.number_input("pH", min_value=3.0, max_value=10.0, value=7.2, step=0.1)
            hrt = st.number_input("HRT (days)", min_value=0.1, max_value=100.0, value=20.0, step=0.5)
            olr = st.number_input("OLR (gVS/L/day)", min_value=0.0, max_value=30.0, value=3.5, step=0.1)

    with col3:
        with st.container(border=True):
            st.markdown('<div class="form-card-head"><div class="form-card-num">03</div><div class="form-card-title"><b>Biological stability indicators</b><small>Capture nitrogen and acidification pressure</small></div></div>', unsafe_allow_html=True)
            isr = st.number_input("ISR (inoculum-to-substrate ratio)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
            tan = st.number_input("TAN (mg/L)", min_value=0.0, max_value=20000.0, value=3500.0, step=50.0, help="Total Ammonia Nitrogen — values above 1500 mg/L may indicate inhibition risk")
            vfa = st.number_input("VFA (mg/L)", min_value=0.0, max_value=30000.0, value=2500.0, step=50.0, help="Volatile Fatty Acids — elevated values may indicate process imbalance")

    st.markdown('<div class="form-footnote">Predictions are decision-support outputs and should be interpreted with laboratory analytics, mass-balance checks, and bioprocess expertise.</div>', unsafe_allow_html=True)
    run = st.form_submit_button("Analyze the bioprocess  →", use_container_width=True)

input_data = {
    "Substrate": substrate,
    "Substrate_Category": substrate_category,
    "Reactor_Type": reactor_type,
    "Temperature_C": temperature,
    "pH": ph,
    "HRT_days": hrt,
    "OLR_gVS_L_day": olr,
    "ISR": isr,
    "TAN_mg_L": tan,
    "VFA_mg_L": vfa,
}

ph_state, temp_state, tan_state, vfa_state = process_profile(input_data)
st.markdown(
    f'''<div class="profile-grid">
    <div class="profile-card"><div class="profile-label">Biological pH window</div><div class="profile-value {ph_state[1]}">{ph_state[0]}</div></div>
    <div class="profile-card"><div class="profile-label">Thermal regime</div><div class="profile-value {temp_state[1]}">{temp_state[0]}</div></div>
    <div class="profile-card"><div class="profile-label">Nitrogen pressure · TAN</div><div class="profile-value {tan_state[1]}">{tan_state[0]}</div></div>
    <div class="profile-card"><div class="profile-label">Acidification pressure · VFA</div><div class="profile-value {vfa_state[1]}">{vfa_state[0]}</div></div>
    </div>''', unsafe_allow_html=True,
)

if run:
    with st.spinner("Running the connected prediction models..."):
        methane_yield = predict_methane_yield(input_data)
        inhibition_probability, inhibition_threshold, risk_detected = predict_inhibition_risk(input_data)
        task3_result = predict_inhibition_type(input_data) if risk_detected else None
    st.session_state["anaerovision_latest_result"] = {
        "methane_yield": methane_yield,
        "inhibition_probability": inhibition_probability,
        "inhibition_threshold": inhibition_threshold,
        "risk_detected": risk_detected,
        "task3_result": task3_result,
        "inputs": input_data.copy(),
    }


# ============================================================
# RESULTS
# ============================================================

st.markdown('<div id="results"></div>', unsafe_allow_html=True)
render_section(
    "Bioprocess decision output",
    "Optimization snapshot",
    "Review predicted conversion performance, process-stability risk, and the most likely inhibition mechanism in one optimization snapshot.",
    "Last submitted scenario",
)

bundle = st.session_state.get("anaerovision_latest_result")
if bundle:
    methane_yield = bundle["methane_yield"]
    inhibition_probability = bundle["inhibition_probability"]
    inhibition_threshold = bundle["inhibition_threshold"]
    risk_detected = bundle["risk_detected"]
    task3_result = bundle["task3_result"]
    submitted = bundle["inputs"]

    if methane_yield >= 450:
        yield_color, yield_note = "safe", "High yield"
    elif methane_yield >= 250:
        yield_color, yield_note = "teal", "Typical range"
    else:
        yield_color, yield_note = "warn", "Below average"

    risk_color = "danger" if risk_detected else "safe"
    risk_label = "Risk detected" if risk_detected else "No risk signal"

    if risk_detected and task3_result:
        confidence_class = {"High":"safe", "Moderate":"warn", "Low":"teal", "Uncertain":"neutral"}.get(task3_result["confidence_level"], "neutral")
        diagnosis_value = f'{task3_result["confidence"]:.3f}'
        diagnosis_label = html.escape(task3_result["predicted_type"])
        confidence_badge = f'{task3_result["confidence_level"]} confidence'
    else:
        confidence_class = "neutral"
        diagnosis_value = "—"
        diagnosis_label = "Not triggered"
        confidence_badge = "Conditional task"

    st.markdown(
        f'''<div class="result-grid">
        <div class="result-card {yield_color}"><div class="result-top"><span class="result-index">TASK 01</span><span class="status-badge {yield_color}">{yield_note}</span></div><div class="result-label">Predicted biomethane yield</div><div class="result-value {yield_color}">{methane_yield:.1f}</div><div class="result-unit">mL CH₄ / gVS</div></div>
        <div class="result-card {risk_color}"><div class="result-top"><span class="result-index">TASK 02</span><span class="status-badge {risk_color}">{risk_label}</span></div><div class="result-label">Process instability probability</div><div class="result-value {risk_color}">{inhibition_probability:.3f}</div><div class="result-unit">Decision threshold {inhibition_threshold:.2f}</div></div>
        <div class="result-card {confidence_class}"><div class="result-top"><span class="result-index">TASK 03</span><span class="status-badge {confidence_class}">{confidence_badge}</span></div><div class="result-label">Mechanism confidence</div><div class="result-value {confidence_class}">{diagnosis_value}</div><div class="result-unit">{diagnosis_label}</div></div>
        </div>''', unsafe_allow_html=True,
    )

    if risk_detected and task3_result:
        interpretation_class = {"High":"safe", "Moderate":"warn", "Low":"teal", "Uncertain":"neutral"}.get(task3_result["confidence_level"], "neutral")
        message = html.escape(task3_result["final_message"].replace("**", ""))
        st.markdown(
            f'''<div class="insight-card {interpretation_class}"><div class="insight-icon">{ICON_BULB}</div><div><div class="insight-label">Bioprocess interpretation · {task3_result["confidence_level"]} confidence</div><div class="insight-copy">{message}</div></div></div>''',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-kicker" style="margin:1.7rem 0 .7rem">Class probability distribution</div>', unsafe_allow_html=True)
        sorted_probs = sorted(task3_result["class_probabilities"].items(), key=lambda item: item[1], reverse=True)
        top_type = sorted_probs[0][0]
        rows = []
        for inhibition_type, probability in sorted_probs:
            top_class = "top" if inhibition_type == top_type else ""
            pct = probability * 100
            rows.append(f'''<div class="prob-row"><div class="prob-name {top_class}">{html.escape(inhibition_type)}</div><div class="prob-track"><div class="prob-fill {top_class}" style="width:{pct:.1f}%"></div></div><div class="prob-pct">{pct:.1f}%</div></div>''')
        st.markdown('<div class="probability-panel">' + ''.join(rows) + '</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'''<div class="insight-card safe"><div class="insight-icon">{ICON_CHECK}</div><div><div class="insight-label">Bioprocess status</div><div class="insight-copy">No inhibition signal was detected, so the conditional mechanism classifier was not activated. The submitted scenario falls within the model's stable operating region.</div></div></div>''',
            unsafe_allow_html=True,
        )

    with st.expander("Review the submitted process conditions"):
        st.markdown(
            f'''<div class="input-summary">
            <div><b>Substrate:</b> {html.escape(str(submitted["Substrate"]))}<br><b>Category:</b> {html.escape(str(submitted["Substrate_Category"]))}<br><b>Reactor:</b> {html.escape(str(submitted["Reactor_Type"]))}<br><b>Temperature:</b> {submitted["Temperature_C"]} °C<br><b>pH:</b> {submitted["pH"]}</div>
            <div><b>HRT:</b> {submitted["HRT_days"]} days<br><b>OLR:</b> {submitted["OLR_gVS_L_day"]} gVS/L/day<br><b>ISR:</b> {submitted["ISR"]}<br><b>TAN:</b> {submitted["TAN_mg_L"]} mg/L<br><b>VFA:</b> {submitted["VFA_mg_L"]} mg/L</div>
            </div>''', unsafe_allow_html=True,
        )
else:
    st.markdown(
        f'''<div class="empty-state"><div class="empty-icon">{ICON_PLAY}</div><b>Your bioprocess workspace is ready</b><p>Configure the reactor state and run the analysis. Biomethane performance, stability risk, and conditional mechanism diagnosis will appear here.</p></div>''',
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f'''<footer class="av-footer">
    <div class="footer-top">
        <div class="footer-brand">{LOGO_SVG}<div><div class="footer-brand-name">Anaero<span>Vision</span></div><div class="footer-brand-sub">Bioprocess optimization intelligence for anaerobic digestion — connecting feedstock characterization, reactor operating windows, and biological stability signals to one decision-support workflow.</div></div></div>
        <div class="footer-links">
            <div class="footer-col"><span class="footer-col-label">Workspace</span><a href="#how-it-works">How it works</a><a href="#predict">Predict</a><a href="#results">Results</a></div>
            <div class="footer-col"><span class="footer-col-label">Model cascade</span><span class="footer-static">01 · Methane yield — tuned ANN</span><span class="footer-static">02 · Inhibition risk — ANN classifier</span><span class="footer-static">03 · Inhibition type — tuned random forest</span></div>
        </div>
    </div>
    <div class="footer-bottom">
        <span>AnaeroVision v1.2 — decision-support tool, not a substitute for laboratory analytics.</span>
        <span>Bioprocess engineering × applied machine learning</span>
    </div>
    </footer>''',
    unsafe_allow_html=True,
)