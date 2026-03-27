import os
import re

REPLACEMENTS = [
    # Terminology
    (r'データポイント', r'データ例'),
    (r'フィーチャー', r'特徴量'),
    (r'グラディエント', r'勾配'),
    (r'サバイバルスキル', r'基礎知識'),
    (r'見つけ出します', r'見つけ出す'),
    
    # Safe verbiage conjugations
    (r'ことができます。', r'可能である。'),
    (r'ことができます', r'可能である'),
    (r'必要となります。', r'必要がある。'),
    (r'必要があります。', r'必要がある。'),
    (r'必要があります', r'必要がある'),
    (r'ではありません。', r'ではない。'),
    (r'ではありません', r'ではない'),
    (r'ありません。', r'ない。'),
    (r'しましょう。', r'しよう。'),
    (r'ましょう。', r'よう。'),
    
    # Specific safe endings (must be grouped before the generic あります etc)
    (r'異なります。', r'異なる。'),
    (r'異なります', r'異なる'),
    (r'成ります。', r'成る。'),
    (r'成ります', r'成る'),
    (r'なります。', r'なる。'),
    (r'なります', r'なる'),
    (r'わかります。', r'わかる。'),
    (r'わかります', r'わかる'),
    (r'あります。', r'ある。'),
    (r'あります', r'ある'),
    (r'おります。', r'おる。'),
    (r'おります', r'おる'),
    (r'行います。', r'行う。'),
    (r'行います', r'行う'),
    (r'使います。', r'使う。'),
    (r'使います', r'使う'),
    (r'用います。', r'用いる。'),
    (r'用います', r'用いる'),
    (r'思います。', r'思う。'),
    (r'思います', r'思う'),
    (r'持ちます。', r'持つ。'),
    (r'持ちます', r'持つ'),
    (r'できます。', r'できる。'),
    (r'できます', r'できる'),
    (r'ています。', r'ている。'),
    (r'ています', r'ている'),
    (r'されます。', r'される。'),
    (r'されます', r'される'),
    (r'行われます。', r'行われる。'),
    (r'行われます', r'行われる'),
    (r'得られます。', r'得られる。'),
    (r'得られます', r'得られる'),
    (r'見られます。', r'見られる。'),
    (r'見られます', r'見られる'),
    (r'続きます。', r'続く。'),
    (r'続きます', r'続く'),
    (r'紹介します。', r'紹介する。'),
    (r'紹介します', r'紹介する'),
    (r'動作します。', r'動作する。'),
    (r'動作します', r'動作する'),

    # su-verb exceptions
    (r'起こします。', r'起こす。'),
    (r'起こします', r'起こす'),
    (r'表します。', r'表す。'),
    (r'表します', r'表す'),
    (r'示します。', r'示す。'),
    (r'示します', r'示す'),
    (r'出します。', r'出す。'),
    (r'出します', r'出す'),
    (r'指します。', r'指す。'),
    (r'指します', r'指す'),
    (r'残します。', r'残す。'),
    (r'残します', r'残す'),
    (r'直します。', r'直す。'),
    (r'直します', r'直す'),
    (r'隠します。', r'隠す。'),
    (r'隠します', r'隠す'),
    (r'返します。', r'返す。'),
    (r'返します', r'返す'),
    (r'回します。', r'回す。'),
    (r'回します', r'回す'),

    (r'しました。', r'した。'),
    (r'します。', r'する。'),
    (r'します', r'する'),
    
    (r'しますが、', r'するが、'),
    (r'しましたが、', r'したが、'),
    (r'ますが、', r'るが、'),
    (r'ですが、', r'であるが、'),
    
    # Fallback endings (safe, low priority)
    (r'ます。', r'る。'),
    (r'です。', r'である。'),
]

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    in_code_block = False
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('{.') and not in_code_block:
            in_code_block = True
            new_lines.append(line)
            continue
            
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            new_lines.append(line)
            continue
            
        if in_code_block or line.startswith(':') or stripped == '':
            new_lines.append(line)
            continue
            
        new_line = line
        for old, new in REPLACEMENTS:
            new_line = re.sub(old, new, new_line)
            
        new_lines.append(new_line)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        for f in os.listdir(target_dir):
            if f.endswith('.md'):
                process_file(os.path.join(target_dir, f))
        print(f"Polishing script execution complete for {target_dir}.")
    else:
        print("Please provide a target directory as an argument.")
        print("Example: python3 jppolish.py chapter_recurrent-neural-networks")
