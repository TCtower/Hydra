from collections import Counter
import nltk.translate.bleu_score as bleu
 
 
def calculate_BLEU(generated_summary, reference_summary, n):
    # Tokenize the generated summary and reference summary
    generated_tokens = generated_summary.split()
    reference_tokens = reference_summary.split()
 
    # Calculate the BLEU score
    weights = [1.0 / n] * n  # Weights for n-gram precision calculation
    bleu_score = bleu.sentence_bleu([reference_tokens], generated_tokens, weights=weights)
 
    return bleu_score
 
 
# Example usage from test_translation_opus_books.json
reference_summary = "Et pourtant, dans les crises d'emportement qui les secouaient, ils lisaient chacun nettement au fond de leur colère, ils devinaient les fureurs de leur être égoïste qui les avaient poussés à l'assassinat pour contenter ses appétits, et qui ne trouvait dans l'assassinat qu'une existence désolée et intolérable."
generated_summary = "Et pourtant, dans les accès de colère qui les agitaient, ils ont tous deux vu clair jusque dans le fond de leur colère. Ils étaient conscients que c'était le violent impulsion de leur nature égoïste qui les avait poussés à tuer pour satisfaire leur désir, et qu'ils n'avaient trouvé que dans l'assassinat, une existence affligeante et insupportable."
n = 2  # Bigram
 
bleu_score = calculate_BLEU(generated_summary, reference_summary, n)
print(f"BLEU-{n} score: {bleu_score}")