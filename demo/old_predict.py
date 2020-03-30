from flask import Flask, render_template, flash, request, url_for, redirect

#Importing the packages we will be using
# Basic Packages
import numpy as np
import pandas as pd
#pd.set_option('display.max_columns', 500)
#np.set_printoptions(suppress=True)

# NLTK Packages
#import nltk
# Use the code below to download the NLTK package, a straightforward GUI should pop up
# nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import joblib

stop_words = stopwords.words('english')
#Adds stuff to our stop words list
stop_words.extend(['.',','])


## This function can improve, simplify. Look into Text Data Lecture
def remove_stopwords(list_of_tokens):
    """
    Removes stopwords
    """
    
    cleaned_tokens = [] 
    
    for token in list_of_tokens: 
        if token in stop_words: continue 
        cleaned_tokens.append(token)
            
    return cleaned_tokens

stop_words = stopwords.words('english')
#Adds stuff to our stop words list
stop_words.extend(['.',','])


## This function can improve, simplify. Look into Text Data Lecture
def remove_stopwords(list_of_tokens):
    """
    Removes stopwords
    """
    
    cleaned_tokens = [] 
    
    for token in list_of_tokens: 
        if token in stop_words: continue 
        cleaned_tokens.append(token)
            
    return cleaned_tokens
def stemmer(list_of_tokens):
    '''
    Takes in an input which is a list of tokens, and spits out a list of stemmed tokens.
    '''
    
    stemmed_tokens_list = []
    
    for i in list_of_tokens:
        
        token = PorterStemmer().stem(i)
        stemmed_tokens_list.append(token)
        
    return stemmed_tokens_list


#from nltk.stem import WordNetLemmatizer

def lemmatizer(list_of_tokens):
    
    lemmatized_tokens_list = []
    
    for i in list_of_tokens: 
        token = WordNetLemmatizer().lemmatize(i)
        lemmatized_tokens_list.append(token)
        
    return lemmatized_tokens_list


def the_untokenizer(token_list):
        '''
        Returns all the tokenized words in the list to one string. 
        Used after the pre processing, such as removing stopwords, and lemmatizing. 
        '''
        return " ".join(token_list)
    
def clean_string(my_string):
    tokenized_list = word_tokenize(my_string)
    removed_stopwords = remove_stopwords(tokenized_list)
    stemmed_words = stemmer(removed_stopwords)
    lemmatized_words = lemmatizer(stemmed_words)
    back_to_string = the_untokenizer(lemmatized_words)
    return back_to_string

app = Flask("genre_prediction")
app.secret_key = "super secret key"

@app.route('/', methods=["GET","POST"])
def predict():

    error = ''
    try:
        if request.method == "POST":
            my_string = request.form['plot']

            train_df = pd.read_csv('train_medians.csv')
            my_model = joblib.load('../4-Modeling/models/my_best_model.pkl')
            my_scaler = joblib.load('../4-Modeling/models/my_best_scaler.pkl')
            my_tfidf = joblib.load('../4-Modeling/models/my_best_tfidf.pkl')
            
            #my_model = joblib.load('../4-Modeling/models/my_1vr_logreg_0.001_lbfgs.pkl')
            #my_scaler = joblib.load('../4-Modeling/models/my_standard_scaler.pkl')
            #my_tfidf = joblib.load('../3-Data-Preprocessing/models/my_tfidf_min20.pkl')
            
            genre_cols = ['action','adventure','animation','biography','comedy','crime','documentary',\
                          'drama','family','fantasy','film-noir','history','horror','music','musical',\
                          'mystery','romance','sci-fi','sport','thriller','war','western']

            feature_cols = ['f_release_year','f_release_month','f_runtime','f_word_count_long','f_imdb_rating',\
                            'f_num_imdb_votes','f_num_user_reviews','f_num_critic_reviews']
                            
                            
            feature_cols_df = pd.DataFrame([[0]*8 ], columns=feature_cols)

            input_tfidf = my_tfidf.transform([clean_string(my_string)])
            input_transformed_df = pd.DataFrame(input_tfidf.toarray(), columns=my_tfidf.get_feature_names())

            input_final = pd.concat([feature_cols_df, input_transformed_df], axis=1)
            
            for col in feature_cols:
                input_final.at[0,col] = train_df[col].median()
            input_final.at[0,'f_word_count_long'] = len(my_string)
            input_final_df = my_scaler.transform(input_final)

            input_pred = my_model.predict_proba(input_final_df)

            df = pd.DataFrame(input_pred, columns=genre_cols).T.sort_values(0, ascending=False)
            output_list = []
            output_dict = {}
            for index, row in df.iterrows():
                if row.values[0] >= 0.2:
                    temp_list = [int(round(row.values[0]*100,0)), index.capitalize()]
                    output_list.append(temp_list)
            return render_template('predict.html', results=output_list, my_string=my_string)
        else:
            return render_template('predict.html')

    except Exception as e:
        flash(e)
        return render_template("predict.html", error = error)  

if __name__ == "__main__":
    #app.debug = True
    app.run()