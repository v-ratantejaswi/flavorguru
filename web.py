import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import csv
import hashlib

st.set_page_config(page_title="FlavorGuru", layout="wide")
@st.cache_data
def load_data():
    data = pd.read_csv("epi_r.csv")
    return data

dataset = load_data()

non_ingredient_columns = [
    'title', 'rating', 'calories', 'protein', 'fat', 'sodium',
    'vegan', 'vegetarian', 'wheat/gluten-free', 'low fat', 'low sugar', 'low sodium',
    'dairy free', 'kid-friendly', 'paleo', 'kosher', 'kosher for passover',
    'peanut free', 'tree nut free', 'soy free', 'no meat, no problem', 'no sugar added', 
    'no-cook', 'grill', 'grill/barbecue', 'boil', 'fry', 'bake', 'roast', 'steam', 
    'sauté', 'stir-fry', 'deep-fry', 'broil', 'marinate', 'simmer', 'poach', 'braise',
    'quick & easy', 'healthy', 'high fiber', 'raw', '#cakeweek', '#wasteless',
    'dinner', 'breakfast', 'lunch', 'brunch', 'appetizer', 'dessert', 'snack', 'drink',
    'valentine\'s day', 'thanksgiving', 'christmas', 'easter', 'halloween', 'super bowl', 
    'fourth of july', 'new year\'s eve', 'new year\'s day', 'hanukkah', 'birthday',
    'wedding', 'anniversary', 'party', 'picnic', 'summer', 'winter', 'fall', 'spring',
    'cocktail party', 'backyard bbq', 'back to school', 'snack week', 'leftovers',
    'bon appétit', 'gourmet', 'house & garden', 'weelicious', 'epi loves the microwave',
    'blender', 'food processor', 'microwave', 'smoker', 'pressure cooker', 'slow cooker', 
    'mixer', 'juicer', 'grill', 'mortar and pestle', 'double boiler', 'coffee grinder',
    'pasta maker', 'candy thermometer',
    'alabama', 'alaska', 'arizona', 'california', 'colorado', 'connecticut', 
    'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 
    'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 
    'minnesota', 'mississippi', 'missouri', 'nebraska', 'new hampshire', 
    'new jersey', 'new mexico', 'new york', 'north carolina', 'ohio', 'oklahoma', 
    'oregon', 'pennsylvania', 'rhode island', 'south carolina', 'tennessee', 
    'texas', 'utah', 'vermont', 'virginia', 'washington', 'west virginia', 'wisconsin',
    'washington, d.c.', 'los angeles', 'san francisco', 'chicago', 'boston', 'atlanta', 
    'dallas', 'miami', 'seattle', 'houston', 'paris', 'london', 
    'israel', 'italy', 'jamaica', 'japan', 'mexico', 'dominican republic', 'france', 
    'germany', 'england', 'spain', 'ireland', 'australia', 'canada', 'egypt', 'beverly hills',
    'brooklyn', 'bulgaria','cambridge','columbus', 'costa mesa', 'florida', 'guam','haiti', 'healdsburg',
    'kansas city', 'lancaster', 'las vegas', 'long beach', 'louisville', 'minneapolis', 'new orleans',
    'pasadena', 'peru', 'philippines', 'pittsburgh','portland', 'providence', 'santa monica', 'st. louis',
    'switzerland', 'westwood', 'windsor',
    'alcoholic', 'non-alcoholic', 'vegetable', 'fruit', 'meat', 'poultry', 'seafood',
    'dairy', 'condiment', 'cocktail', 'sandwich', 'candy', 'edible gift', 'cake', 'pescatarian', 'side', 
    'sugar conscious', 'kidney friendly', 'low cal', 'soup/stew', 'leafy green', 'condiment/spread',
    '#cakeweek', '#wasteless', '22-minute meals', '3-ingredient recipes', 
    '30 days of groceries', 'advance prep required', 'anthony bourdain', 'bastille day',
    'bon app��tit',  'camping', 'chill', 'christmas eve', 'cook like a diner', 'cookbook critic',
    'diwali','dorie greenspan', 'emeril lagasse', 'engagement party', 'entertaining', 'epi + ushg',
    'family reunion', "father's day", 'flaming hot summer', 'frankenrecipe', 'freeze/chill', 'freezer food', 
    'friendsgiving', 'frozen dessert', 'fruit juice', 'game','graduation', 'grand marnier', 'harpercollins',
    'hollywood',"hors d'oeuvre", 'hot drink', 'house cocktail', 'ice cream', 'ice cream machine', 
    'iced coffee', 'iced tea','kentucky derby','kitchen olympics','kwanzaa', 'labor day', 
    'low carb', 'low cholesterol', 'low/no sugar', 'lunar new year', 'mardi gras',"mother's day", 
    'nancy silverton',  'oktoberfest', 'one-pot meal', 'oscars','pacific palisades', 'pan-fry',
    'passover', 'persian new year', 'pizza', 'poker/game night', 'quick and healthy', 'ramadan',
    'sandwich theory', 'sangria', "st. patrick's day", 'tailgating', 'tested & improved', 'triple sec', 
    'tropical fruit', 'yonkers', 'cookbooks'
]


ingredient_columns = [col for col in dataset.columns if col not in non_ingredient_columns]
ingredient_sums = dataset[ingredient_columns].sum().sort_values(ascending=False)
dataset['combined_text'] = dataset['title'] + " " + dataset[non_ingredient_columns].apply(lambda row: ' '.join(row.index[row == 1]), axis=1)

transactions = dataset[ingredient_columns].apply(lambda x: list(x.index[x == 1]), axis=1).tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
recipe_titles = dataset['title'].dropna().unique().tolist()
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)


def association_rule_recommendations(ingredient_list, rules):
    recommendations = set()
    for ingredient in ingredient_list:
        consequents = rules[rules['antecedents'] == {ingredient}]['consequents']
        for items in consequents:
            recommendations.update(items)
    return [item for item in recommendations if item not in ingredient_list]


def recipe_recommendations_based_on_ingredients(selected_ingredients, rules, dataset):
    recommended_ingredients = association_rule_recommendations(selected_ingredients, rules)
    scores = []
    for index, row in dataset.iterrows():
        score = sum([row[ingredient] for ingredient in recommended_ingredients if ingredient in row])
        scores.append((index, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_recipe_indices = [index for index, score in scores[:10]]
    return dataset['title'].iloc[top_recipe_indices]


def recipe_recommendations_based_on_tags(matching_tags, rules, dataset):
    recommended_tags = association_rule_recommendations(matching_tags, rules)
    scores = []
    for index, row in dataset.iterrows():
        score = sum([row[tag] for tag in recommended_tags if tag in row])
        scores.append((index, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_recipe_indices = [index for index, score in scores[:10]]
    return dataset['title'].iloc[top_recipe_indices]

def modified_ingredient_based_recommendations(dataset, ingredient_columns, rules):
    try:
        num_ingredients = int(input("How many ingredients would you like to choose from? "))
        top_ingredients = dataset[ingredient_columns].sum().sort_values(ascending=False).head(num_ingredients)
        print("Top ingredients to choose from:")
        for i, ingredient in enumerate(top_ingredients.index, 1):
            print(f"{i}. {ingredient}")
        selected_ingredients = []
        for _ in range(3):
            ingredient_input = input("Enter ingredient name: ").strip()
            if ingredient_input not in top_ingredients.index:
                raise ValueError(f"'{ingredient_input}' is not in the top ingredients list.")
            selected_ingredients.append(ingredient_input)
        recommendations = recipe_recommendations_based_on_ingredients(selected_ingredients, rules, dataset)
        print("Top 10 recipe recommendations:")
        for title in recommendations:
            print(title)
    except ValueError as ve:
        print(f"Invalid input: {ve}")

def extract_ingredients_from_recipe(recipe_title, dataset, ingredient_columns):
    if recipe_title not in dataset['title'].values:
        print(f"The recipe '{recipe_title}' does not exist in the dataset.")
        return None
    recipe_row = dataset[dataset['title'] == recipe_title].iloc[0]
    return [col for col in ingredient_columns if recipe_row[col] == 1]

def get_recommendations_based_on_recipe(recipe_title, rules, dataset, ingredient_columns):
    ingredients = extract_ingredients_from_recipe(recipe_title, dataset, ingredient_columns)
    if ingredients is None:
        return None
    return recipe_recommendations_based_on_ingredients(ingredients, rules, dataset)

def user_recipe_based_recommendation(recipe_title, rules, dataset, ingredient_columns):
    # recipe_title += " "  # Adding space to match the dataset format
    recommendations = get_recommendations_based_on_recipe(recipe_title, rules, dataset, ingredient_columns)
    
    if recommendations is not None:
        print("Top 10 recipe recommendations:")
        return recommendations
    else:
        print("Recipe not found or no recommendations available.")

def recommend_recipes_on_recipe(title, dataset, cosine_sim):
    if title not in dataset['title'].values:
        print(f"Recipe '{title}' not found.")
        return None

    idx = dataset.index[dataset['title'] == title].tolist()[0]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    recipe_indices = [i[0] for i in sim_scores]

    return dataset['title'].iloc[recipe_indices]


def recommend_based_on_preference(pref, lower_b, upper_b, recipe_count, dataset):
    preference = pref
    if preference not in ['calories', 'rating']:
        print("Invalid preference. Please choose 'calories' or 'rating'.")
        return

    try:
        lower_bound = float(lower_b)
        upper_bound = float(upper_b)
        num_recipes = int(recipe_count)
        filtered_recipes = dataset[(dataset[preference] >= lower_bound) & (dataset[preference] <= upper_bound)]
        return filtered_recipes.nlargest(num_recipes, preference)['title'].tolist()
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return []

def process_user_query(user_query, dataset, rules, ingredient_columns, non_ingredient_columns):
    tokens = word_tokenize(user_query.lower())
    matching_ingredients = [ingredient for ingredient in ingredient_columns if any(token in ingredient for token in tokens)]
    matching_tags = [tag for tag in non_ingredient_columns if any(token in tag for token in tokens)]
    if matching_ingredients:
        ingredient_recommendations = recipe_recommendations_based_on_ingredients(matching_ingredients, rules, dataset)
    else:
        ingredient_recommendations = []
    if matching_tags:
        tag_recommendations = recipe_recommendations_based_on_tags(matching_tags, rules, dataset)
    else:
        tag_recommendations = []
    print(ingredient_recommendations)
    print(tag_recommendations)
    combined_recommendations = pd.concat([ingredient_recommendations, tag_recommendations]).drop_duplicates().tolist()


    return combined_recommendations[:10] if combined_recommendations else "No matching recipes found based on your preferences."


def load_users(filename='users.csv'):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['username', 'password', 'email'])
        df.to_csv(filename, index=False)
        return df


def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_user(username, password_hash):
    users_df = load_users() 
    return not users_df[(users_df['username'] == username) & (users_df['password'] == password_hash)].empty


def register_user(username, password, email):
    new_user = pd.DataFrame([[username, hash_password(password), email]], columns=['username', 'password', 'email'])
    users_df = load_users()
    updated_users_df = pd.concat([users_df, new_user], ignore_index=True)
    updated_users_df.to_csv('users.csv', index=False)


def user_auth():
    if st.session_state.get('loggedin', False):
        st.sidebar.write(f"Logged in as {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            st.session_state.loggedin = False
            st.session_state.username = ''
            st.experimental_rerun()
    else:
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login"):
            if check_user(username, hash_password(password)):
                st.session_state.loggedin = True
                st.session_state.username = username
                st.experimental_rerun()
            else:
                st.sidebar.error("Invalid username or password")

        if st.sidebar.button("Register"):
            st.session_state.register = True
            st.experimental_rerun()


def registration_page():
    st.title("Register New User")
    with st.form("Register User"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        email = st.text_input("Email")
        submit = st.form_submit_button("Register")
        
        if submit:
            register_user(username, password, email)
            st.success("Registered Successfully")
            st.session_state.register = False
            st.experimental_rerun()
    
    if st.button("Back to Login"):
        st.session_state.register = False
        st.experimental_rerun()


if 'users_df' not in st.session_state:
    st.session_state.users_df = load_users()

if 'loggedin' not in st.session_state:
    st.session_state['loggedin'] = False

if 'username' not in st.session_state:
    st.session_state['username'] = ''

if 'register' not in st.session_state:
    st.session_state['register'] = False


def main_app():
    with st.sidebar:
        st.title("FlavorGuru")
        user_choice = st.radio("Choose an option:", ["Ingredient-based", "Recipe-based", "User Preferences", "Query-based"])


    if user_choice == "Ingredient-based":
        st.header("Ingredient-based Recommendations")
        ingredients = st.multiselect("Select ingredients:", ingredient_columns)
        if st.button("Get Recommendations"):
            recommendations = recipe_recommendations_based_on_ingredients(ingredients, rules, dataset)
            st.subheader("Recommendations:")
            for rec in recommendations:
                st.write(rec)

   
    elif user_choice == "Recipe-based":
        st.header("Recipe-based Recommendations")
        recipe_title = st.selectbox("Select or enter a recipe name:", recipe_titles)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(dataset['combined_text'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        if st.button("Get Recommendations"):
            recommendations = recommend_recipes_on_recipe(recipe_title,dataset, cosine_sim )
            st.subheader("Recommendations:")
            if len(recommendations) > 0:
                for rec in recommendations:
                    st.write(rec)
            else:
                st.write("No recommendations found for this recipe.")

    elif user_choice == "User Preferences":
        st.header("Preferences-based Recommendations")
        preference = st.selectbox("Choose preference", ["Calories", "Rating"])
        lower_bound = st.number_input(f"Enter lower bound for {preference}:", min_value=0.0, value=100.0)
        upper_bound = st.number_input(f"Enter upper bound for {preference}:", min_value=0.0, value=500.0)
        num_recipes = st.number_input("How many recipes do you want to see?", min_value=1, max_value=10, value=5)
        if st.button("Get Recommendations"):
            recommendations = recommend_based_on_preference(preference.lower(), lower_bound, upper_bound, num_recipes, dataset)
            st.subheader("Recommendations:")
            for rec in recommendations:
                st.write(rec)

    elif user_choice == "Query-based":
        st.header("Query Based Recommendations")
        user_query = st.text_input("Enter your food preferences:")
        if st.button("Get Recomendations"):
            recommendations = process_user_query(user_query, dataset, rules, ingredient_columns, non_ingredient_columns)
            if isinstance(recommendations, str):
                st.write(recommendations)
            else:
                st.subheader("Recommended recipes:")
                for title in recommendations:
                    st.write(title)



if __name__ == '__main__':
    st.title('Welcome to FlavorGuru')
    user_auth()

    if st.session_state.get('register', False):
        registration_page()
    elif st.session_state.get('loggedin', False):
        main_app()
    else:
        st.image('home.webp', width=1000)
        st.write("Please login to access the app")
        if st.button("Register New User"):
            st.session_state.register = True
            st.experimental_rerun()