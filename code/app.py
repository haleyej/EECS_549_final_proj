import pickle
import csv
import streamlit as st 


def load_front_end_data():
    info_dict = {}
    with open("../data/front_end_data.csv", "r") as f: 
        reader = csv.reader(f)
        for line in reader:
            info_dict[int(line[0])] = line[1:]


def app():
    with open("../system_components/l2r", "rb") as f: 
        l2r = pickle.load(f)

    with open("../system_components/bm25", "wb") as f:
        bm25 = pickle.load(f)

    info_dict = load_front_end_data()
    
    st.title("Search r/askreddit")

    query = st.text_input("Search: ",  key = "query")

    search_type = st.radio(
        "Search Method:",
        ['Learning to Rank', 'BM25 (Baseline)'],
        index=0,
        key="search_type_input",
        horizontal=True,
    )

    if st.button("SEARCH"):
        if search_type == "Learning to Rank":
            output_data = l2r.query(query)
        elif search_type == "BM25 (Baseline)":
            output_data = bm25.query(query)
        else:
            output_data = []
        show_results(output_data, info_dict)

    st.markdown("<p style='text-align: center;'> Haley Johnson | EECS 549 F23 </p>", unsafe_allow_html=True)




def show_results(results, info_dict):
    aa = 0
    st.info(f"Showing results for: {len(results)}")

    st.info(f"Note that some posts have been deleted")

    N_cards_per_row = 3
    for n_row, (docid, score) in enumerate(results):
        i = n_row % N_cards_per_row
        if i == 0:
            st.write("---")
            cols = st.columns(N_cards_per_row, gap="large")
        # Draw the card
        with cols[n_row % N_cards_per_row]:
            info = info_dict.get(docid, [])
            if info != []:
                continue
            q = info[0].lower()
            link = info[1]
            post_karma = info[2]

            st.caption(f"{q}")
            st.markdown(f"Post Karma: {post_karma}")
            st.markdown(f"{link}")
                       
        aa += 1

    if aa == 0:
        st.info("No results found. Please try again.")
    else:
        st.info(f"Results shown for: {aa}")

def main():
    import pickle
    import csv
    app()

if __name__ == '__main__':
    main()