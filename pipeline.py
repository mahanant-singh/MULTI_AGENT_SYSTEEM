from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain


def banner(text: str) -> None:
    print("\n" + "=" * 50)
    print(text)
    print("=" * 50)


def run_research_pipeline(topic: str) -> dict:
    state = {}

    # Step 1 - Search
    banner("Step 1 - Search agent is working...")
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = search_result["messages"][-1].content
    print("\nSearch results:\n", state["search_results"])

    # Step 2 - Reader
    banner("Step 2 - Reader agent is scraping top resources...")
    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results']}"
        )]
    })
    state["scraped_content"] = reader_result["messages"][-1].content
    print("\nScraped content:\n", state["scraped_content"])

    # Step 3 - Writer
    banner("Step 3 - Writer is drafting the report...")
    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )
    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined,
    })
    print("\nFinal report:\n", state["report"])

    # Step 4 - Critic
    banner("Step 4 - Critic is reviewing the report...")
    state["feedback"] = critic_chain.invoke({"report": state["report"]})
    print("\nCritic feedback:\n", state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ").strip()
    if not topic:
        print("Topic empty hai. Program band ho raha hai.")
    else:
        try:
            run_research_pipeline(topic)
        except Exception as e:
            print(f"\nPipeline fail ho gaya: {e}")