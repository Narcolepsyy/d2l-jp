import os
import re
import string

def sanitize_slug(text):
    # Remove "英: " if present
    text = text.replace('英: ', '').strip()
    # slugify
    text = text.lower()
    valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
    slug = ''.join(c for c in text if c in valid_chars)
    slug = slug.replace(' ', '-').replace('_', '-')
    # remove duplicate dashes
    slug = re.sub(r'-+', '-', slug)
    return slug.strip('-')

def extract_terms():
    terms = {}
    
    # Regex to match *Japanese Term* (English Term) or **Japanese Term** (English Term)
    # The English term should mostly be ascii letters, spaces, hyphens, and maybe "英:" prefix
    pattern = re.compile(r'(\*\*|\*)([^*]+)\1（(?:英:\s*)?([a-zA-Z0-9\s\-]+)）')
    
    for root, dirs, files in os.walk('.'):
        if 'chapter_glossary' in root or '.git' in root or '.agent' in root or 'venv' in root or '_build' in root:
            continue
            
        if not os.path.basename(root).startswith('chapter_'):
            continue
            
        for file in files:
            if file.endswith('.md') and not file.startswith('.'):
                file_path = os.path.join(root, file)
                chapter_name = os.path.basename(root)
                file_name = file
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # We will process paragraph by paragraph to get the context
                paragraphs = content.split('\n\n')
                for p in paragraphs:
                    matches = pattern.findall(p)
                    for match in matches:
                        _, ja_term, en_term = match
                        ja_term = ja_term.strip()
                        en_term = en_term.replace('英:', '').strip()
                        slug = sanitize_slug(en_term)
                        
                        if not slug or len(slug) < 2:
                            continue
                            
                        # Keep the paragraph as definition, strip markdown links if any, or keep them.
                        # We'll keep the paragraph to provide context.
                        definition = p.strip()
                        
                        # Only keep reasonable size definitions to avoid thin content (SEO)
                        if len(definition) > 50 and len(definition) < 1000:
                            if slug not in terms:
                                terms[slug] = {
                                    'ja': ja_term,
                                    'en': en_term,
                                    'def': definition,
                                    'ref': f"../{chapter_name}/{file_name}"
                                }
                                
    return terms

def generate_glossary(terms):
    out_dir = 'chapter_glossary'
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate individual pages
    index_links = []
    
    # Sort terms by Japanese reading (naively by character for now) or English slug
    sorted_slugs = sorted(terms.keys())
    
    for slug in sorted_slugs:
        data = terms[slug]
        page_path = os.path.join(out_dir, f"{slug}.md")
        
        md_content = f"""# {data['ja']} ({data['en']})
:label:`sec_glossary_{slug}`

## 定義 (Definition)

{data['def']}

## 参照 (Reference)

この用語の詳細な文脈については Dive into Deep Learning の対応する章を参照してください:
- [元章で読む]({data['ref']})
"""
        with open(page_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        index_links.append(f"   {slug}")
        
    # Generate index.md
    index_content = f"""# ディープラーニング用語集 (Deep Learning Glossary)
:label:`chap_glossary`

これは、Dive into Deep Learning (D2L) で使用される重要なディープラーニングおよび機械学習の概念をまとめた用語集です。
各用語ページには、定義と関連する章へのリンクが含まれています。

```toc
:maxdepth: 1

{chr(10).join(index_links)}
```
"""
    with open(os.path.join(out_dir, 'index.md'), 'w', encoding='utf-8') as f:
        f.write(index_content)
        
    print(f"Generated glossary with {len(terms)} terms.")

if __name__ == '__main__':
    terms = extract_terms()
    generate_glossary(terms)
