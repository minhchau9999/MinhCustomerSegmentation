import re
import pandas as pd

# Hàm kiểm tra từ có phải là từ tiếng Việt "sạch"
def is_valid_vietnamese(word):
    vietnamese_chars = (
        "a-zA-Z0-9_"
        "àáạảãâầấậẩẫăằắặẳẵ"
        "èéẹẻẽêềếệểễ"
        "ìíịỉĩ"
        "òóọỏõôồốộổỗơờớợởỡ"
        "ùúụủũưừứựửữ"
        "ỳýỵỷỹ"
        "đ"
        "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ"
        "ÈÉẸẺẼÊỀẾỆỂỄ"
        "ÌÍỊỈĨ"
        "ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
        "ÙÚỤỦŨƯỪỨỰỬỮ"
        "ỲÝỴỶỸ"
        "Đ"
    )
    pattern = f'^[{vietnamese_chars}]+$'
    return re.match(pattern, word) is not None


# Hàm xử lý một mô tả
def filter_vietnamese_words(text):
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
    # Remove emojis and other non-text characters
    # This regex pattern matches emoji and other special characters
    text = re.sub(r'[^\w\s,.!?;:()[\]{}\'\"\/\\-]', '', text)
    
    # Original Vietnamese word filtering logic
    words = text.split()
    vietnamese_words = []
    
    for word in words:
        # Keep only words with Vietnamese characters or basic alphanumeric
        if re.search(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', word.lower()) or re.match(r'^[a-zA-Z0-9]+$', word):
            vietnamese_words.append(word)
    
    return ' '.join(vietnamese_words)

# Hàm tiền xử lý dũ liệu cho Gensim
from typing import Optional
def data_preprocessing_for_gensim(text, stop_words = None, remove_number: Optional[bool] = None, remove_special_chars: Optional[bool] = None):
    if remove_number:
        text_re = [[re.sub('[0-9]+','', e) for e in text] for text in text]
    if remove_special_chars:
        text_re = [[t.lower() for t in text if not t in ['', ' ', ',', '.', '...', '-',':', ';', '?', '%', '(', ')', '+', '/', "'", '&','⭐','💢','🏘','☎','📖','🌱','❤','📞','🎯','💥','⛔']] for text in  text_re] 
    if stop_words:
        text_re = [[t for t in text if not t in stop_words] for text in text_re] # stopword
    return text_re

# Hàm lấy sản phẩm đề xuất dựa trên Gensim
def get_recommendations_gensim(similarity_index, df, tfidf, lsi_model, dictionary, query=None, product_id=None, nums=10, stop_words=None):
    
    """
    Get product recommendations using Gensim's similarity index
    
    Args:
        similarity_index: Gensim similarity index
        df: DataFrame containing product information
        tfidf: TF-IDF model
        lsi_model: LSI model
        dictionary: Gensim dictionary
        query: Text query for search-based recommendations (for use case 2)
        product_id: ID of the product to get recommendations for (for use case 1)
        nums: Number of recommendations to return
        
    Returns:
        DataFrame with recommended products
    """
    # Use case 1: User selects a product ID
    if product_id is not None:
        # Get the index of the product
        idx = df.index[df['product_id'] == product_id][0]
        
        # Get the document vector for the product
        doc_vector = df['content_processed'][idx]
        
        # Get the sub_category of the selected product
        selected_sub_category = df.iloc[idx]['sub_category'] if 'sub_category' in df.columns else None
        
        # Convert to bag of words
        bow_vector = dictionary.doc2bow(doc_vector)
        
        # For use case 1, we'll exclude the selected product from results
        exclude_idx = idx
        exclude_product_id = product_id
        
    # Use case 2: User searches with a text query
    elif query is not None:
        # Process the query text (assuming same preprocessing as content_processed)
        processed_query = preprocess_text(query,stop_words=stop_words)  
        
        # Convert to bag of words
        bow_vector = dictionary.doc2bow(processed_query)
        
        # For use case 2, we don't need to exclude any specific product
        exclude_idx = None
        exclude_product_id = None
        selected_sub_category = None
        
    else:
        raise ValueError("Either product_id or query must be provided")
    
    # Transform to TF-IDF and LSI space
    tfidf_vector = tfidf[bow_vector]
    lsi_vector = lsi_model[tfidf_vector]
    
    # Get similarities
    sims = similarity_index[lsi_vector]
    
    # Sort the similarities
    sim_scores = list(enumerate(sims))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top N*2 similar products (excluding the product itself if needed)
    # We get more than needed to allow for filtering and prioritization
    if exclude_idx is not None:
        sim_scores = [s for s in sim_scores if s[0] != exclude_idx][:nums*2]
    else:
        sim_scores = sim_scores[:nums*2]
    
    # Get the indices of the similar products
    product_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]
    
    # Create a DataFrame with the similar products
    result = df.iloc[product_indices].copy()
    result['similarity_score'] = similarity_scores
    
    # Double-check to ensure the input product_id is not in the results
    if exclude_product_id is not None:
        result = result[result['product_id'] != exclude_product_id]
    
    # For use case 1 (product_id), prioritize same sub_category and higher rating
    if product_id is not None and selected_sub_category is not None and 'sub_category' in result.columns:
        # Sort by similarity score first, then by sub_category, then by rating (if available)
        sort_columns = ['similarity_score']
        sort_ascending = [False]
        
        # Add sub_category as a secondary sort criterion
        # Create a temporary column to prioritize same sub_category
        result['same_category'] = (result['sub_category'] == selected_sub_category).astype(int)
        sort_columns.append('same_category')
        sort_ascending.append(False)  # Same category first
        
        # Add rating as a tertiary sort criterion if available
        if 'rating' in result.columns:
            sort_columns.append('rating')
            sort_ascending.append(False)  # Higher ratings first
        
        # Sort by the specified columns
        result = result.sort_values(sort_columns, ascending=sort_ascending)
        
        # Drop the temporary same_category column
        result = result.drop('same_category', axis=1)
    # For use case 2 (query-based search), prioritize higher rating
    elif query is not None and 'rating' in result.columns:
        # Sort by similarity score first, then by rating
        result = result.sort_values(['similarity_score', 'rating'], ascending=[False, False])
    
    # Select the columns to return
    columns_to_return = ['product_id', 'product_name', 'similarity_score']
    if 'rating' in result.columns:
        columns_to_return.insert(2, 'rating')
    if 'sub_category' in result.columns:
        columns_to_return.insert(2, 'sub_category')
    
    return result[columns_to_return].head(nums)

# Helper function to preprocess text queries
def preprocess_text(text, stop_words=None):
    """
    Preprocess text query using the same steps as for content_processed
    
    Args:
        text: Raw text query
        stop_words: List of stop words to remove
        
    Returns:
        List of processed tokens
    """
    # Use the same preprocessing as the training data
    # First convert the text to a list of tokens
    tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
    
    # Then apply the same data_preprocessing_for_gensim function
    # Note: data_preprocessing_for_gensim expects a list of lists
    processed_tokens = data_preprocessing_for_gensim(
        [tokens], 
        stop_words=stop_words,
        remove_number=True,
        remove_special_chars=True
    )
    
    # The function returns a list of lists, so we take the first element
    return processed_tokens[0] if processed_tokens else []

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations_cosine(tfidf_matrix, df, query=None, product_id=None, nums=10, vectorizer=None):
    """
    Get product recommendations using cosine similarity
    
    Args:
        tfidf_matrix: TF-IDF matrix of product descriptions
        df: DataFrame containing product information
        query: Text query for search-based recommendations (for use case 2)
        product_id: ID of the product to get recommendations for (for use case 1)
        nums: Number of recommendations to return
        vectorizer: The TfidfVectorizer used to create the tfidf_matrix (needed for query-based search)
        
    Returns:
        DataFrame with recommended products
    """
    # Use case 1: User selects a product ID
    if product_id is not None:
        # Find the index of the product with the given ID
        idx = df.index[df['product_id'] == product_id][0]
        
        # Get the TF-IDF vector for this product
        product_vector = tfidf_matrix[idx:idx+1]
        
        # Get the sub_category of the selected product
        selected_sub_category = df.iloc[idx]['sub_category'] if 'sub_category' in df.columns else None
        
        # For use case 1, we'll exclude the selected product from results
        exclude_idx = idx
        exclude_product_id = product_id
        
    # Use case 2: User searches with a text query
    elif query is not None:
        if vectorizer is None:
            raise ValueError("Vectorizer must be provided for query-based recommendations")
            
        # Process the query text and transform to TF-IDF vector
        # We need to preprocess the query the same way as the original data
        processed_query = preprocess_text(query)
        
        # Join the processed tokens back into a string for the vectorizer
        query_text = ' '.join(processed_query)
        
        # Transform the query to a TF-IDF vector
        # We need to use transform (not fit_transform) to use the same vocabulary
        product_vector = vectorizer.transform([query_text])
        
        # For use case 2, we don't need to exclude any specific product
        exclude_idx = None
        exclude_product_id = None
        selected_sub_category = None
        
    else:
        raise ValueError("Either product_id or query must be provided")
    
    # Calculate similarity with all products
    sim_scores = cosine_similarity(product_vector, tfidf_matrix).flatten()
    
    # Create list of (index, similarity score) tuples
    sim_scores_with_indices = list(enumerate(sim_scores))
    
    # Sort by similarity score
    sim_scores_with_indices = sorted(sim_scores_with_indices, key=lambda x: x[1], reverse=True)
    
    # Get the top N*2 similar products (excluding the product itself if needed)
    # We get more than needed to allow for filtering and prioritization
    if exclude_idx is not None:
        sim_scores_with_indices = [s for s in sim_scores_with_indices if s[0] != exclude_idx][:nums*2]
    else:
        sim_scores_with_indices = sim_scores_with_indices[:nums*2]
    
    # Get indices of similar products and their similarity scores
    product_indices = [i[0] for i in sim_scores_with_indices]
    similarity_scores = [i[1] for i in sim_scores_with_indices]
    
    # Create a DataFrame with the similar products
    result = df.iloc[product_indices].copy()
    result['similarity_score'] = similarity_scores
    
    # Double-check to ensure the input product_id is not in the results
    if exclude_product_id is not None:
        result = result[result['product_id'] != exclude_product_id]
    
    # For use case 1 (product_id), prioritize same sub_category and higher rating
    if product_id is not None and selected_sub_category is not None and 'sub_category' in result.columns:
        # Sort by similarity score first, then by sub_category, then by rating (if available)
        sort_columns = ['similarity_score']
        sort_ascending = [False]
        
        # Add sub_category as a secondary sort criterion
        # Create a temporary column to prioritize same sub_category
        result['same_category'] = (result['sub_category'] == selected_sub_category).astype(int)
        sort_columns.append('same_category')
        sort_ascending.append(False)  # Same category first
        
        # Add rating as a tertiary sort criterion if available
        if 'rating' in result.columns:
            sort_columns.append('rating')
            sort_ascending.append(False)  # Higher ratings first
        
        # Sort by the specified columns
        result = result.sort_values(sort_columns, ascending=sort_ascending)
        
        # Drop the temporary same_category column
        result = result.drop('same_category', axis=1)
    # For use case 2 (query-based search), prioritize higher rating
    elif query is not None and 'rating' in result.columns:
        # Sort by similarity score first, then by rating
        result = result.sort_values(['similarity_score', 'rating'], ascending=[False, False])
    
    # Select the columns to return
    columns_to_return = ['product_id', 'product_name', 'similarity_score']
    if 'rating' in result.columns:
        columns_to_return.insert(2, 'rating')
    if 'sub_category' in result.columns:
        columns_to_return.insert(2, 'sub_category')
    
    return result[columns_to_return].head(nums)

def get_recommendations_surprise(df_productid,full_product_df, surprise, user_id, nums=10):
    # Create predictions for all products for this user
    #copy the df first
    df_copy = df_productid.copy()
    df_copy['Score_Prediction'] = df_copy['product_id'].apply(
        lambda x: surprise.predict(int(user_id), x).est
    )
    
    # Sort by prediction score and get top N
    recommendations = df_copy.sort_values(
        by=['Score_Prediction'], 
        ascending=False
    ).drop_duplicates().head(nums)

    recommendations['product_name'] = recommendations['product_id'].apply(lambda x: full_product_df[full_product_df['product_id'] == x]['product_name'].values[0])
    recommendations['sub_category'] = recommendations['product_id'].apply(lambda x: full_product_df[full_product_df['product_id'] == x]['sub_category'].values[0])
    recommendations['rating'] = recommendations['product_id'].apply(lambda x: full_product_df[full_product_df['product_id'] == x]['rating'].values[0])
    
    return recommendations


    
    