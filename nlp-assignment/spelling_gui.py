#!/usr/bin/env python3
"""
GUI Spelling Correction Application
Meets assignment requirements with 500-character editor and advanced NLP techniques
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import sys
import os
import threading
import time

# Add the script's directory to sys.path to ensure imports work from any location
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'spelling'))

from spelling.src.assets import load_vocab, load_word_freq
from spelling.src.candidates import load_symspell_words, CandidateGenerator  
from spelling.src.advanced_rank import AdvancedRanker
from spelling.src.text import normalize, tokenize
from spelling.src.detect import NonWordDetector, RealWordDetector, MixedErrorHandler
from rapidfuzz.distance import Levenshtein

class SpellingCorrectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced NLP Spelling Correction System")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        # Load spelling correction components
        self.load_system()
        
        # Create GUI components
        self.create_widgets()
        
        # Track misspelled words
        self.misspelled_words = {}
        self.word_suggestions = {}
        self.word_error_types = {}  # Track whether each word is 'misspelled' (non-word) or 'realword'
        
    def load_system(self):
        """Load the high-performance spelling correction system"""
        try:
            print("Loading advanced spelling correction system...")
            self.vocab = load_vocab()
            self.word_freqs = load_word_freq()
            self.symspell_words = load_symspell_words()
            
            # Enhanced: Load enhanced vocabulary with technical terms
            from spelling.src.assets import load_enhanced_vocab
            self.enhanced_vocab = load_enhanced_vocab(self.vocab, include_technical=True)
            print(f"Enhanced vocabulary loaded: {len(self.enhanced_vocab)} words (including technical terms)")
            
            # Use enhanced vocab for all components
            self.generator = CandidateGenerator(self.symspell_words, self.enhanced_vocab, radius=2, use_enhanced_vocab=True)
            self.ranker = AdvancedRanker(self.word_freqs)
            self.nonword_detector = NonWordDetector(self.enhanced_vocab)
            self.realword_detector = RealWordDetector(self.enhanced_vocab, self.word_freqs)
            # Enhanced: Add mixed error handler for complex cases
            self.mixed_error_handler = MixedErrorHandler(
                self.nonword_detector, self.realword_detector, self.generator
            )
            print("✅ Enhanced system loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load spelling system: {e}")
            
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main title
        title = tk.Label(self.root, text="Advanced NLP Spelling Correction System", 
                        font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#2c3e50")
        title.pack(pady=10)
        
        # Create main frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Text editor frame
        editor_frame = tk.LabelFrame(main_frame, text="Text Editor (500 characters max)", 
                                   font=("Arial", 12, "bold"), bg="#f0f0f0", fg="#34495e")
        editor_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Text editor with 500 character limit
        self.text_editor = scrolledtext.ScrolledText(
            editor_frame, height=12, width=70, wrap=tk.WORD,
            font=("Consolas", 11), bg="#ffffff", fg="#2c3e50",
            insertbackground="#e74c3c", selectbackground="#3498db"
        )
        self.text_editor.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bind events
        self.text_editor.bind('<KeyRelease>', self.on_text_change)
        self.text_editor.bind('<Button-1>', self.on_word_click)
        self.text_editor.bind('<KeyPress>', self.check_char_limit)
        
        # Character counter
        self.char_count_label = tk.Label(main_frame, text="Characters: 0/500", 
                                       font=("Arial", 10), bg="#f0f0f0", fg="#7f8c8d")
        self.char_count_label.pack(anchor=tk.E)
        
        # Control buttons frame
        control_frame = tk.Frame(main_frame, bg="#f0f0f0")
        control_frame.pack(fill=tk.X, pady=10)
        
        # Check spelling button
        self.check_button = tk.Button(
            control_frame, text="🔍 Check Spelling", font=("Arial", 12, "bold"),
            bg="#3498db", fg="white", command=self.check_spelling,
            relief=tk.RAISED, borderwidth=2, padx=20, pady=5
        )
        self.check_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear highlighting button
        self.clear_button = tk.Button(
            control_frame, text="✖ Clear Highlighting", font=("Arial", 12),
            bg="#e74c3c", fg="white", command=self.clear_highlighting,
            relief=tk.RAISED, borderwidth=2, padx=20, pady=5
        )
        self.clear_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Show vocabulary button
        self.vocab_button = tk.Button(
            control_frame, text="📚 Show Vocabulary", font=("Arial", 12),
            bg="#27ae60", fg="white", command=self.show_vocabulary,
            relief=tk.RAISED, borderwidth=2, padx=20, pady=5
        )
        self.vocab_button.pack(side=tk.LEFT)
        
        # Results frame
        results_frame = tk.LabelFrame(main_frame, text="Spelling Analysis", 
                                    font=("Arial", 12, "bold"), bg="#f0f0f0", fg="#34495e")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(
            results_frame, height=8, width=70, wrap=tk.WORD,
            font=("Consolas", 10), bg="#ecf0f1", fg="#2c3e50", state=tk.DISABLED
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Load text and click 'Check Spelling'")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                            font=("Arial", 10), bg="#34495e", fg="white", anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Configure text tags for highlighting
        self.text_editor.tag_configure("misspelled", background="#ffcccc", foreground="#c0392b")
        self.text_editor.tag_configure("realword", background="#fff3cd", foreground="#856404")
        
    def check_char_limit(self, event):
        """Check character limit before allowing input"""
        current_length = len(self.text_editor.get("1.0", tk.END)) - 1
        if current_length >= 500 and event.keysym not in ['BackSpace', 'Delete', 'Left', 'Right', 'Up', 'Down']:
            return "break"
            
    def on_text_change(self, event):
        """Update character count when text changes"""
        current_length = len(self.text_editor.get("1.0", tk.END)) - 1
        self.char_count_label.config(text=f"Characters: {current_length}/500")
        
        # Change color if approaching limit
        if current_length > 450:
            self.char_count_label.config(fg="#e74c3c")
        elif current_length > 400:
            self.char_count_label.config(fg="#f39c12")
        else:
            self.char_count_label.config(fg="#7f8c8d")
            
    def check_spelling(self):
        """FAST spelling check optimized for GUI performance"""
        text = self.text_editor.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to check.")
            return
            
        # Disable button to prevent multiple simultaneous checks
        self.check_button.config(state=tk.DISABLED, text="🔄 Analyzing...")
        self.status_var.set("Running fast spelling analysis...")
        self.root.update()
        
        # Run analysis in background thread to prevent GUI freezing
        def run_analysis():
            try:
                start_time = time.time()
                
                # Clear previous highlighting (must be done in main thread)
                self.root.after(0, self.clear_highlighting)
                
                # Tokenize text - use tokenize() without normalize() to preserve case
                # This matches how test scripts use .split() which also preserves case
                # The detector internally handles case conversion
                tokens = tokenize(text)
                
                # FAST detection - only check alphabetic tokens
                nonword_errors = []
                realword_errors = []
                compound_errors = []
                
                # Build confusion pairs set ONCE for O(1) lookups
                confusion_set = set(self.realword_detector.confusion_pairs.keys())
                
                # Single pass through tokens for non-word errors (FAST)
                for i, token in enumerate(tokens):
                    if token.isalpha():
                        token_lower = token.lower()
                        
                        # Fast non-word check (vocabulary lookup only)
                        # Use enhanced_vocab to match test scripts
                        if token_lower not in self.enhanced_vocab:
                            nonword_errors.append((i, token_lower))
                
                # Use the full context-aware real-word detection
                # This is more accurate than just checking confusion_set membership
                real_word_results = self.realword_detector.detect(tokens)
                flags = real_word_results['flags']
                compound_errors = real_word_results['compound_errors']
                
                for i, (token, is_error) in enumerate(zip(tokens, flags)):
                    if is_error and token.isalpha():
                        token_lower = token.lower()
                        # Don't double-count non-word errors
                        if token_lower in self.vocab:
                            realword_errors.append((i, token_lower))
                
                elapsed = time.time() - start_time
                
                # Pre-generate suggestions for all errors (batch processing with context)
                self.misspelled_words = {}
                self.word_suggestions = {}
                self.word_error_types = {}  # Reset error type tracking
                
                # Create a mapping of words to their positions for context
                # Also track the error type for each word
                for i, word in nonword_errors:
                    if word not in self.word_suggestions:
                        self.word_suggestions[word] = self.get_suggestions_fast(word, tokens, i)
                    self.word_error_types[word] = 'misspelled'  # Non-word error
                
                for i, word in realword_errors:
                    if word not in self.word_suggestions:
                        self.word_suggestions[word] = self.get_suggestions_fast(word, tokens, i)
                    self.word_error_types[word] = 'realword'  # Real-word error
                
                results = []
                results.append("=== FAST SPELLING ANALYSIS ===")
                results.append("")
                results.append(f"Analysis time: {elapsed:.2f} seconds")
                results.append(f"Total tokens analyzed: {len([t for t in tokens if t.isalpha()])}")
                results.append("")
                results.append(f"Non-word errors found: {len(nonword_errors)}")
                results.append(f"Real-word errors found: {len(realword_errors)}")
                if compound_errors:
                    results.append(f"Compound word issues: {len(compound_errors)}")
                results.append("")
                
                # Process non-word errors
                if nonword_errors:
                    results.append("NON-WORD ERRORS (words that don't exist):")
                    for i, word in nonword_errors:
                        suggestions = self.word_suggestions[word]
                        # Schedule highlighting in main thread
                        self.root.after(0, lambda w=word: self.highlight_word(w, "misspelled"))
                        
                        results.append(f"• '{word}':")
                        for j, (suggestion, edit_dist) in enumerate(suggestions[:5], 1):
                            results.append(f"  {j}. {suggestion} (edit distance: {edit_dist})")
                        results.append("")
                
                # Process real-word errors
                if realword_errors:
                    results.append("REAL-WORD ERRORS (words used in wrong context):")
                    for i, word in realword_errors:
                        suggestions = self.word_suggestions[word]
                        # Schedule highlighting in main thread
                        self.root.after(0, lambda w=word: self.highlight_word(w, "realword"))
                        
                        results.append(f"• '{word}' at position {i+1} (check context):")
                        for j, (suggestion, edit_dist) in enumerate(suggestions[:3], 1):
                            results.append(f"  {j}. Consider: {suggestion}")
                        results.append("")
                
                # Process compound word errors
                if compound_errors:
                    results.append("COMPOUND WORD ISSUES (possible split words):")
                    for idx, msg in compound_errors:
                        results.append(f"• {msg}")
                    results.append("")
                
                if not nonword_errors and not realword_errors:
                    results.append("✅ No spelling errors detected! Text looks good.")
                
                results.append("")
                results.append("=== CLICK ON HIGHLIGHTED WORDS FOR SUGGESTIONS ===")
                
                # Update GUI in main thread
                def update_results():
                    self.results_text.config(state=tk.NORMAL)
                    self.results_text.delete("1.0", tk.END)
                    self.results_text.insert("1.0", "\n".join(results))
                    self.results_text.config(state=tk.DISABLED)
                    
                    total_errors = len(nonword_errors) + len(realword_errors) + len(compound_errors)
                    self.status_var.set(f"⚡ Analysis complete in {elapsed:.2f}s - Found {total_errors} potential errors")
                    self.check_button.config(state=tk.NORMAL, text="🔍 Check Spelling")
                
                self.root.after(0, update_results)
                
            except Exception as e:
                def show_error():
                    messagebox.showerror("Error", f"Spelling check failed: {e}")
                    self.status_var.set("Error occurred during analysis")
                    self.check_button.config(state=tk.NORMAL, text="🔍 Check Spelling")
                
                self.root.after(0, show_error)
        
        # Start analysis in background thread
        threading.Thread(target=run_analysis, daemon=True).start()
    
    def get_suggestions(self, word):
        """Get spelling suggestions with edit distances"""
        try:
            # Generate candidates using ultra-aggressive mode
            candidates = self.generator.generate(word, use_symspell=True, aggressive=True, ultra=True)
            if not candidates:
                return [("No suggestions found", 0)]
            
            # Rank candidates
            sentence = ['the', word, 'is']
            suggestions = self.ranker.suggest(candidates, word, sentence, 1, top_k=10, 
                                           synthetic_mode=True, ultra_mode=True)
            
            # Calculate edit distances
            suggestions_with_distance = []
            for suggestion in suggestions[:8]:  # Limit to top 8
                edit_dist = Levenshtein.distance(word, suggestion)
                suggestions_with_distance.append((suggestion, edit_dist))
            
            return suggestions_with_distance
            
        except Exception as e:
            return [("Error generating suggestions", 0)]
    
    def get_suggestions_fast(self, word, context_tokens=None, index=-1):
        """FAST suggestion generation optimized for GUI speed with context"""
        try:
            word_lower = word.lower()
            
            # ===== REAL-WORD ERROR HANDLING =====
            # For real-word errors, prioritize confusion pairs FIRST - they are the correct answers!
            if word_lower in self.realword_detector.confusion_pairs:
                # Get the confusion pair alternatives (these are the CORRECT suggestions)
                alternatives = self.realword_detector.confusion_pairs[word_lower]
                
                # Build result list with confusion pairs at the top
                result = []
                for alt in alternatives:
                    edit_dist = Levenshtein.distance(word_lower, alt)
                    result.append((alt, edit_dist))
                
                # Add a few more general candidates if needed (but confusion pairs stay on top)
                if len(result) < 5:
                    candidates = self.generator.generate(word_lower, use_symspell=True, aggressive=False, ultra=False)
                    for cand in candidates[:5]:
                        if cand not in alternatives:
                            edit_dist = Levenshtein.distance(word_lower, cand)
                            result.append((cand, edit_dist))
                
                return result[:8]
            
            # ===== NON-WORD ERROR HANDLING =====
            # For non-word errors, use edit-distance based ranking
            candidates = self.generator.generate(word_lower, use_symspell=True, aggressive=False, ultra=False, 
                                               context_tokens=context_tokens, index=index)
            
            if not candidates:
                return [("No suggestions found", 0)]
            
            # IMPROVED RANKING: Edit distance is PRIMARY, frequency is SECONDARY
            import math
            candidates_scored = []
            for candidate in candidates[:20]:  # Consider more candidates
                freq = self.word_freqs.get(candidate, 1)
                edit_dist = Levenshtein.distance(word_lower, candidate)
                
                freq_score = math.log(freq + 1) * 10  # Log scale to reduce frequency dominance
                edit_penalty = edit_dist * 100  # Heavy penalty for each edit
                score = freq_score - edit_penalty
                
                candidates_scored.append((candidate, edit_dist, score))
            
            # Sort by edit distance first (ascending), then by score (descending)
            candidates_scored.sort(key=lambda x: (x[1], -x[2]))
            
            # Return top suggestions with edit distances
            return [(cand, dist) for cand, dist, score in candidates_scored[:8]]
            
        except Exception as e:
            return [("Error generating suggestions", 0)]
    
    def highlight_word(self, word, tag):
        """Highlight misspelled words in the text editor (case-insensitive)"""
        text_content = self.text_editor.get("1.0", tk.END)
        start = "1.0"
        
        while True:
            # Use case-insensitive search with nocase option
            pos = self.text_editor.search(word, start, tk.END, nocase=True)
            if not pos:
                break
            
            end_pos = f"{pos}+{len(word)}c"
            self.text_editor.tag_add(tag, pos, end_pos)
            start = end_pos
    
    def on_word_click(self, event):
        """Handle clicking on words to show suggestions"""
        # Get clicked position
        index = self.text_editor.index(tk.CURRENT)
        
        # Get the word at the clicked position
        line, col = map(int, index.split('.'))
        text_content = self.text_editor.get(f"{line}.0", f"{line}.end")
        
        if col < len(text_content):
            # Find word boundaries
            start = col
            while start > 0 and text_content[start-1].isalpha():
                start -= 1
            end = col
            while end < len(text_content) and text_content[end].isalpha():
                end += 1
            
            if start < end:
                word = text_content[start:end].lower()
                
                # Show suggestions if it's a misspelled word
                if word in self.word_suggestions:
                    self.show_word_suggestions(word, event)
    
    def show_word_suggestions(self, word, event):
        """Show suggestions popup for a clicked word"""
        suggestions = self.word_suggestions.get(word, [])
        if not suggestions:
            return
        
        # Create popup menu
        popup = tk.Menu(self.root, tearoff=0)
        popup.add_command(label=f"Suggestions for '{word}':", state=tk.DISABLED)
        popup.add_separator()
        
        for suggestion, edit_dist in suggestions[:6]:
            popup.add_command(
                label=f"{suggestion} (distance: {edit_dist})",
                command=lambda s=suggestion, w=word: self.replace_word(w, s)
            )
        
        popup.add_separator()
        popup.add_command(label="Ignore", command=lambda: None)
        
        try:
            popup.tk_popup(event.x_root, event.y_root)
        finally:
            popup.grab_release()
    
    def replace_word(self, old_word, new_word):
        """Replace a word in the text editor (case-insensitive replacement)"""
        text_content = self.text_editor.get("1.0", tk.END)
        
        # Find and replace the word (case-insensitive)
        import re
        # Create a case-insensitive pattern that matches whole words
        pattern = r'\b' + re.escape(old_word) + r'\b'
        
        def replace_func(match):
            original = match.group()
            # Preserve original capitalization pattern
            if original.isupper():
                return new_word.upper()
            elif original.istitle():
                return new_word.capitalize()
            else:
                return new_word.lower()
        
        updated_text = re.sub(pattern, replace_func, text_content, flags=re.IGNORECASE)
        
        self.text_editor.delete("1.0", tk.END)
        self.text_editor.insert("1.0", updated_text)
        
        # Remove ONLY this word's highlighting (don't clear all highlights)
        self.text_editor.tag_remove("misspelled", "1.0", tk.END)
        self.text_editor.tag_remove("realword", "1.0", tk.END)
        
        # Remove from word suggestions and error type tracking
        if old_word in self.word_suggestions:
            del self.word_suggestions[old_word]
        if old_word in self.word_error_types:
            del self.word_error_types[old_word]
        
        # Re-highlight remaining words with their CORRECT error type
        for word in self.word_suggestions.keys():
            error_type = self.word_error_types.get(word, 'misspelled')  # Default to misspelled if not found
            self.highlight_word(word, error_type)
        
        # Update character count only
        self.on_text_change(None)
        
        self.status_var.set(f"Replaced '{old_word}' with '{new_word}'")
    
    def clear_highlighting(self):
        """Clear all highlighting from the text editor"""
        self.text_editor.tag_remove("misspelled", "1.0", tk.END)
        self.text_editor.tag_remove("realword", "1.0", tk.END)
        self.misspelled_words = {}
        self.word_suggestions = {}
        self.word_error_types = {}  # Also clear error type tracking
    
    def show_vocabulary(self):
        """Show vocabulary browser window"""
        vocab_window = tk.Toplevel(self.root)
        vocab_window.title("Vocabulary Browser")
        vocab_window.geometry("600x500")
        vocab_window.configure(bg="#f0f0f0")
        
        # Search frame
        search_frame = tk.Frame(vocab_window, bg="#f0f0f0")
        search_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(search_frame, text="Search vocabulary:", font=("Arial", 12), 
                bg="#f0f0f0").pack(side=tk.LEFT)
        
        search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=search_var, font=("Arial", 12))
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Vocabulary listbox with scrollbar
        list_frame = tk.Frame(vocab_window)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        vocab_listbox = tk.Listbox(list_frame, font=("Consolas", 10))
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=vocab_listbox.yview)
        vocab_listbox.configure(yscrollcommand=scrollbar.set)
        
        vocab_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate vocabulary (sorted)
        sorted_vocab = sorted(self.vocab)
        for word in sorted_vocab:
            freq = self.word_freqs.get(word, 0)
            vocab_listbox.insert(tk.END, f"{word} (freq: {freq})")
        
        # Search functionality
        def search_vocab(*args):
            search_term = search_var.get().lower()
            vocab_listbox.delete(0, tk.END)
            
            if search_term:
                matching_words = [w for w in sorted_vocab if search_term in w.lower()]
            else:
                matching_words = sorted_vocab
            
            for word in matching_words[:1000]:  # Limit to 1000 results
                freq = self.word_freqs.get(word, 0)
                vocab_listbox.insert(tk.END, f"{word} (freq: {freq})")
        
        search_var.trace('w', search_vocab)
        
        # Info label
        info_label = tk.Label(vocab_window, 
                            text=f"Total vocabulary: {len(self.vocab):,} words", 
                            font=("Arial", 10), bg="#f0f0f0", fg="#7f8c8d")
        info_label.pack(pady=(0, 10))

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = SpellingCorrectionGUI(root)
    
    # Add sample text
    sample_text = """This is a sampl text with som mispelled words. The algorith used for speling correction employs advansed NLP tecniques including minimum edit distanse and bigram analysis."""
    app.text_editor.insert("1.0", sample_text)
    app.on_text_change(None)
    
    root.mainloop()

if __name__ == "__main__":
    main()