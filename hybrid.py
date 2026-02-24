from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridLayer:
    def __init__(self, product_df):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.product_df = product_df.drop_duplicates("product_id")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.product_df['product_name'])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        self.product_ids = self.product_df['product_id'].tolist()

    def get_content_score(self, product_id, reference_product_id):
        if product_id in self.product_ids and reference_product_id in self.product_ids:
            idx1 = self.product_ids.index(product_id)
            idx2 = self.product_ids.index(reference_product_id)
            return self.similarity_matrix[idx1][idx2]
        return 0

    def apply_hybrid(self, predictions, reference_product_id):
        blended = []

        for product_id, score, context_bonus, emotion_bonus in predictions:
            content_score = self.get_content_score(product_id, reference_product_id)
            final_score = 0.7 * score + 0.3 * content_score
            blended.append((product_id, final_score, context_bonus, emotion_bonus, content_score))

        blended.sort(key=lambda x: x[1], reverse=True)
        return blended