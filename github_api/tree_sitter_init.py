import os
import sys
from tree_sitter import Language

if len(sys.argv) != 3:
    print("Usage: python tree_sitter_init.py <output_library_path> <java_grammar_dir>")
    sys.exit(1)

output_path = os.path.abspath(sys.argv[1])
java_grammar_dir = os.path.abspath(sys.argv[2])

# Clone grammar if not already available
if not os.path.exists(java_grammar_dir):
    print(f"📥 Cloning tree-sitter-java into {java_grammar_dir}...")
    os.system(f"git clone https://github.com/tree-sitter/tree-sitter-java {java_grammar_dir}")

print(f"🛠  Building Tree-sitter language library at {output_path}...")
Language.build_library(
    # Store the compiled shared object
    output_path,
    # Include the Java grammar
    [java_grammar_dir],
)

print(f"✅ Built Tree-sitter library at {output_path}")
