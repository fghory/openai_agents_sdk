from __future__ import annotations

import asyncio
import agentops
import time, os
from agents import Runner, custom_span, gen_trace_id, trace
from .agents.planner_agent import WebSearchItem, WebSearchPlan, planner_agent
from .agents.search_agent import search_agent
from .agents.writer_agent import ReportData, writer_agent


AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")
agentops.init(AGENTOPS_API_KEY)

async def async_main(topic:str):
    trace_id = gen_trace_id()
    with trace("Research trace", trace_id=trace_id):
        search_plan =  await Runner.run(planner_agent,f"Query: {topic}")
        items=search_plan.final_output.searches
        #print(items)
        result=[item.query for item in items]
        #print(result)

        search_tasks = [Runner.run(search_agent, item.query) for item in items]
        search_results = await asyncio.gather(*search_tasks)

        # Extract final_output from each RunResult and join them
        combined = "\n".join(res.final_output for res in search_results)
        # print(combined)
        writing = Runner.run_streamed(
            writer_agent,
            combined,
        )
        
        async for event in writing.stream_events():
            pass

        final=writing.final_output_as(ReportData)
        with open("topic.md", "w", encoding="utf-8") as f:
            f.write(final.markdown_report)
        #print(final)

def main():
    topic = input("What would you like to research? ")
    asyncio.run(async_main(topic))

if __name__ == "__main__":
    main()
