import re

html = """
<div class="citation" id="id4" role="doc-biblioentry">
<span class="label"><span class="fn-bracket">[</span>AHMJ+14<span class="fn-bracket">]</span></span>
<p>Ossama Abdel-Hamid, Abdel-Rahman Mohamed, Hui Jiang, Li Deng, Gerald Penn, and Dong Yu. Convolutional neural networks for speech recognition. <em>IEEE/ACM Transactions on Audio, Speech, and Language Processing</em>, 22(10):1533–1545, 2014.</p>
</div>
<div class="citation" id="id12" role="doc-biblioentry">
<span class="label"><span class="fn-bracket">[</span>Aba16<span class="fn-bracket">]</span></span>
<p>Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, and et al. TensorFlow: a system for large-scale machine learning. In <em>12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16)</em>, 265–283. 2016.</p>
</div>
<div class="citation" id="id93" role="doc-biblioentry">
<span class="label"><span class="fn-bracket">[</span>GKXS18<span class="fn-bracket">]</span></span>
<p>Akhilesh Gotmare, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. A closer look at deep learning heuristics: learning rate restarts, warmup and distillation. <em>ArXiv:1810.13243</em>, 2018.</p>
</div>
"""

def replace_citation(m):
    original_html = m.group(0)
    p_match = re.search(r"<p>(.*?)</p>", original_html, re.DOTALL)
    if not p_match: return original_html
    
    p_text = re.sub(r"<[^>]+>", "", p_match.group(1)).strip()
    
    # Extract year (4 digits near the end)
    year_match = re.search(r"(\d{4})\.$", p_text)
    year = year_match.group(1) if year_match else "n.d."
    
    # Extract first author last name
    # The format is "First Last, Second, and Third"
    first_author_match = re.match(r"^([^,]+)(?:,|$)", p_text)
    if first_author_match:
        first_author_full = first_author_match.group(1)
        last_name = first_author_full.split()[-1]
    else:
        last_name = "Unknown"
        
    # Check for multiple authors
    if " and " in p_text or ", " in p_text.split(". ")[0]:
        label_text = f"{last_name} et al., {year}"
    else:
        label_text = f"{last_name}, {year}"
        
    return re.sub(r'<span class="label">.*?</span>', f'<span class="label">{label_text}</span>', original_html)

print(re.sub(r'<div class="citation"[^>]*>.*?</div>', replace_citation, html, flags=re.DOTALL))
