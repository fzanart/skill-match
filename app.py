import gradio as gr
from src.main import SkillMatch

resume = SkillMatch()
job_ad = SkillMatch()


def compare_skills():
    resume_skills = set(resume.skills)
    job_skills = set(job_ad.skills)

    missing_skills = job_skills - resume_skills

    return f"Missing skills: {', '.join(missing_skills)}"


with gr.Blocks() as demo:
    gr.Markdown("# Resume and Job Ad Skill Comparison")

    with gr.Row():
        with gr.Column():
            resume_input = gr.Textbox(
                # value=sample_resume,
                placeholder="Enter your resume here...",
                label="Resume Text",
            )
            resume_output = gr.HighlightedText(
                label="Analyzed Resume",
                show_legend=True,
                combine_adjacent=True,
                show_inline_category=False,
            )

        with gr.Column():
            job_ad_input = gr.Textbox(
                # value=sample_job_ad,
                placeholder="Enter the job ad here...",
                label="Job Ad Text",
            )
            job_ad_output = gr.HighlightedText(
                label="Analyzed Job Ad",
                show_legend=True,
                combine_adjacent=True,
                show_inline_category=False,
            )

    analyze_button = gr.Button("Analyze and Compare")
    missing_skills_output = gr.Textbox(label="Missing Skills")

    analyze_button.click(
        fn=lambda r, j: [resume.ner(r), job_ad.ner(j), compare_skills()],
        inputs=[resume_input, job_ad_input],
        outputs=[resume_output, job_ad_output, missing_skills_output],
    )

demo.launch()
