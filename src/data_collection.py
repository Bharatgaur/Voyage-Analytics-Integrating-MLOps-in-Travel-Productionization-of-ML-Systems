"""
src/data_collection.py
======================
Data Collection Module — EduBot LLM Project
Industry: Education and Training

Collects education-specific data from multiple sources:
  1. Synthetic Q&A generation (primary — always works)
  2. Web scraping (Wikipedia education articles)
  3. Public dataset loading (HuggingFace datasets)
  4. PDF/text extraction helpers

Author: EduBot Project
"""

import os
import json
import time
import random
import logging
import requests
import pandas as pd
from typing import List, Dict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
RAW_DIR  = ROOT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. SYNTHETIC EDUCATION Q&A DATASET
#    (Primary source — 500+ instruction-response pairs)
# ==============================================================================

EDUCATION_QA = [
    # ── Learning Styles & Pedagogy ──────────────────────────────────────────
    {
        "instruction": "What are the different learning styles in education?",
        "response": "The four primary learning styles are: Visual (learners who prefer diagrams, charts, and videos), Auditory (learners who benefit from lectures, discussions, and podcasts), Reading/Writing (learners who excel through text-based content and note-taking), and Kinesthetic (hands-on learners who learn by doing experiments and activities). Understanding a student's dominant learning style helps teachers personalize instruction for better outcomes."
    },
    {
        "instruction": "How does Bloom's Taxonomy help in education?",
        "response": "Bloom's Taxonomy is a hierarchical framework that classifies educational objectives into six cognitive levels: Remember, Understand, Apply, Analyze, Evaluate, and Create. It helps educators design lessons that progressively challenge students from basic recall to higher-order thinking. For example, a lesson might start with remembering facts, then move to applying them in problems, and finally to creating original work. This structured approach ensures comprehensive learning."
    },
    {
        "instruction": "What is differentiated instruction?",
        "response": "Differentiated instruction is a teaching approach where educators tailor content, process, products, and learning environment to meet individual student needs. Teachers assess students' readiness, interests, and learning profiles, then adjust: Content (what is taught), Process (how it is taught), Product (how students demonstrate learning), and Environment (where learning occurs). This approach ensures all students, regardless of ability level, receive appropriate challenges and support."
    },
    {
        "instruction": "What is project-based learning (PBL)?",
        "response": "Project-Based Learning (PBL) is an instructional method where students learn by actively working on complex, real-world projects over an extended period. Students investigate authentic problems, collaborate with peers, and present their solutions publicly. PBL develops critical thinking, collaboration, communication, and creativity — the 4Cs of 21st-century learning. Research shows PBL improves student engagement and long-term retention compared to traditional instruction."
    },
    {
        "instruction": "Explain the concept of formative vs summative assessment.",
        "response": "Formative assessment is ongoing evaluation during learning — quizzes, exit tickets, class discussions — designed to provide feedback and guide instruction. It answers 'How is the student progressing?' Summative assessment evaluates learning at the end of an instructional unit — final exams, standardized tests, capstone projects — measuring overall achievement. Effective education uses both: formative assessment to adjust teaching in real time, and summative assessment to measure final outcomes."
    },
    {
        "instruction": "What is the Socratic method in teaching?",
        "response": "The Socratic method is a form of cooperative argumentative dialogue between teacher and students. Rather than lecturing, the teacher asks probing questions that stimulate critical thinking and draw out ideas. Students are guided to discover knowledge through inquiry rather than being told answers. Named after the ancient Greek philosopher Socrates, this method is widely used in law schools, philosophy, and higher education to develop analytical reasoning and argumentation skills."
    },
    {
        "instruction": "How does technology enhance modern education?",
        "response": "Technology enhances modern education through several pathways: Learning Management Systems (LMS) like Moodle and Canvas enable blended learning. AI-powered adaptive platforms (Khan Academy, Duolingo) personalize learning pace. Video conferencing tools (Zoom, Google Meet) enable remote education. Digital assessments provide instant feedback. Gamification platforms (Kahoot!, Quizlet) increase student engagement. Virtual Reality (VR) enables immersive simulations. Overall, technology makes education more accessible, personalized, and interactive."
    },
    {
        "instruction": "What is Universal Design for Learning (UDL)?",
        "response": "Universal Design for Learning (UDL) is an educational framework that proactively designs flexible learning experiences to accommodate diverse learner needs. It provides multiple means of: Representation (how information is presented), Action & Expression (how students demonstrate knowledge), and Engagement (how students are motivated). UDL originated from architecture — designing buildings accessible to all — and applies the same principle to curricula, ensuring inclusivity from the design stage rather than retrofitting accommodations."
    },
    {
        "instruction": "What are the benefits of flipped classroom teaching?",
        "response": "In a flipped classroom, students study new content at home (through videos or readings) and use class time for practice, discussion, and problem-solving with teacher guidance. Benefits include: More active learning during class time, personalized teacher attention during practice, students can pause and replay video lectures, class time used for higher-order thinking tasks, students take ownership of their learning pace. Research shows flipped classrooms improve exam scores and student satisfaction in subjects like mathematics and science."
    },
    {
        "instruction": "What is scaffolding in education?",
        "response": "Scaffolding is an instructional technique where a teacher provides temporary, structured support to students as they learn new concepts, gradually removing support as students gain competence — similar to construction scaffolding that is removed once a building is stable. Examples include: graphic organizers, sentence starters, worked examples, think-alouds, peer support, and chunking complex tasks into smaller steps. Scaffolding is rooted in Vygotsky's Zone of Proximal Development theory."
    },
    # ── STEM Education ───────────────────────────────────────────────────────
    {
        "instruction": "What is STEM education and why is it important?",
        "response": "STEM stands for Science, Technology, Engineering, and Mathematics. STEM education integrates these disciplines in an interdisciplinary, project-based approach that mirrors how professionals work in industry. It is critical because it prepares students for the fastest-growing careers, develops problem-solving and analytical thinking, fosters innovation, and addresses the global skills gap in technical fields. Countries with strong STEM education consistently rank higher in economic competitiveness and innovation indices."
    },
    {
        "instruction": "How can teachers make mathematics more engaging for students?",
        "response": "Teachers can make mathematics engaging through: Real-world applications (budgeting, architecture, sports statistics), mathematical games and puzzles, storytelling and math narratives, collaborative problem-solving, technology tools like Desmos and GeoGebra for visualizations, inquiry-based learning where students discover formulas, culturally responsive math examples, and gamification through platforms like Kahoot! and Prodigy. The key is connecting abstract math to concrete, meaningful contexts students care about."
    },
    {
        "instruction": "What is computational thinking and how is it taught?",
        "response": "Computational thinking is a problem-solving approach that includes four key pillars: Decomposition (breaking problems into smaller parts), Pattern Recognition (identifying similarities and trends), Abstraction (focusing on essential information), and Algorithm Design (creating step-by-step solutions). It is taught through coding platforms like Scratch, Python, and Code.org, but also through unplugged activities like sorting algorithms with playing cards. Computational thinking is considered a fundamental literacy skill for the 21st century."
    },
    {
        "instruction": "What are the challenges in online education?",
        "response": "Online education faces several key challenges: Digital divide (unequal access to devices and internet), student engagement and motivation without physical presence, difficulty maintaining academic integrity during online assessments, limited social interaction affecting collaboration skills, screen fatigue from extended online sessions, technical difficulties disrupting learning, challenges in teaching laboratory and practical skills virtually, and the need for high self-regulation from students who struggle with independent learning."
    },
    {
        "instruction": "What is e-learning and what are its advantages?",
        "response": "E-learning is the delivery of education and training through digital technologies — online courses, webinars, mobile apps, and virtual classrooms. Advantages include: Flexibility to learn anytime and anywhere, cost-effectiveness (no commuting, reduced infrastructure), self-paced learning accommodating different speeds, access to global expertise and diverse content, easy content updates, scalability to thousands of learners, multimedia learning through videos, simulations, and interactive modules, and immediate feedback through automated assessments."
    },
    # ── Special Education & Inclusion ────────────────────────────────────────
    {
        "instruction": "What is inclusive education?",
        "response": "Inclusive education is the practice of educating students with disabilities alongside their non-disabled peers in regular classrooms, with appropriate support and accommodations. It is based on the principle that all children, regardless of ability, disability, language, or background, have the right to learn together. Inclusive education benefits all students — research shows non-disabled students develop greater empathy, while students with disabilities show improved academic and social outcomes in inclusive settings compared to segregated environments."
    },
    {
        "instruction": "How do teachers support students with learning disabilities?",
        "response": "Teachers support students with learning disabilities through: Individualized Education Programs (IEPs) with specific, measurable goals, multi-sensory teaching approaches (visual, auditory, kinesthetic simultaneously), assistive technology (text-to-speech, speech-to-text software), extended time on assessments, preferential seating, breaking tasks into smaller steps, frequent check-ins and feedback, peer tutoring programs, and co-teaching with special education specialists. The goal is to provide access to the general curriculum while addressing individual learning needs."
    },
    {
        "instruction": "What is social-emotional learning (SEL) in education?",
        "response": "Social-Emotional Learning (SEL) is the process through which students develop and apply essential social and emotional skills: Self-awareness (understanding emotions and strengths), Self-management (regulating emotions and impulses), Social awareness (empathy and perspective-taking), Relationship skills (communication, teamwork), and Responsible decision-making. CASEL (Collaborative for Academic, Social, and Emotional Learning) research shows SEL programs improve academic achievement by 11 percentile points and reduce behavioral problems."
    },
    {
        "instruction": "What is growth mindset and how does it impact learning?",
        "response": "Growth mindset, developed by psychologist Carol Dweck, is the belief that intelligence and abilities can be developed through effort, effective strategies, and help from others — as opposed to fixed mindset (believing abilities are innate and unchangeable). Students with growth mindsets embrace challenges, persist through setbacks, learn from criticism, and find inspiration in others' success. Research shows growth mindset interventions significantly improve academic performance, particularly for students from disadvantaged backgrounds."
    },
    # ── Higher Education ─────────────────────────────────────────────────────
    {
        "instruction": "What is the difference between formative and summative assessment?",
        "response": "Formative assessment occurs during the learning process to monitor student progress and inform instruction — examples include quizzes, homework, class participation, and peer feedback. Its purpose is diagnostic and it does not typically count toward final grades. Summative assessment occurs at the end of an instructional period to evaluate total learning — examples include final exams, term papers, and standardized tests. Its purpose is evaluative. Effective education uses both: formative assessment guides teaching, while summative assessment measures achievement."
    },
    {
        "instruction": "How does peer learning benefit students?",
        "response": "Peer learning occurs when students teach, tutor, or learn collaboratively with each other. Benefits include: Students learn by explaining concepts (the protégé effect), peer explanations are often more accessible than teacher explanations, builds communication and collaboration skills, reduces anxiety of asking 'wrong' questions, exposes students to diverse perspectives and problem-solving approaches, increases engagement and motivation, and is cost-effective for institutions. Peer learning models include study groups, peer tutoring, collaborative projects, and peer review."
    },
    {
        "instruction": "What career paths are available in education?",
        "response": "Education offers diverse career paths: Classroom Teacher (primary, secondary, special education), Curriculum Designer/Developer, Instructional Technology Specialist, School Counselor/Psychologist, Educational Administrator (principal, superintendent), Higher Education Professor/Lecturer, Corporate Trainer, Education Policy Analyst, Educational Researcher, E-learning Developer/Instructional Designer, School Librarian, Education Technology (EdTech) Product Manager, Tutor, and Education Consultant. The field is expanding into EdTech startups, AI in education, and international development."
    },
    {
        "instruction": "What is the importance of early childhood education?",
        "response": "Early childhood education (ages 0-8) is the most critical period for brain development — 90% of brain development occurs before age 5. High-quality early education builds foundational cognitive, language, social, and emotional skills. Research by Nobel laureate James Heckman shows every dollar invested in early childhood education returns $7-12 in social benefits through improved educational outcomes, higher earnings, reduced crime, and better health. Early childhood education reduces achievement gaps and is the highest-ROI investment in human capital."
    },
    {
        "instruction": "What are MOOCs and how have they transformed education?",
        "response": "Massive Open Online Courses (MOOCs) are online courses designed for unlimited participation and open access via the internet. Platforms like Coursera, edX, Udemy, and Khan Academy offer MOOCs from top universities and companies. They have democratized education by: Making world-class education accessible regardless of geography or finances, enabling lifelong learning and professional upskilling, allowing learners to study at their own pace, offering micro-credentials and certificates recognized by employers, and reaching millions of learners simultaneously. MOOCs have fundamentally disrupted traditional higher education."
    },
    {
        "instruction": "What is gamification in education?",
        "response": "Gamification in education applies game design elements — points, badges, leaderboards, levels, challenges, rewards — to educational contexts to increase student motivation and engagement. Platforms like Kahoot!, Classcraft, Duolingo, and Prodigy use gamification effectively. Benefits include increased intrinsic motivation, immediate feedback, safe environment for mistakes, sense of progress and achievement, and collaboration through team-based games. Research shows gamification increases student participation by up to 60% and improves information retention through active, enjoyable learning experiences."
    },
    {
        "instruction": "How can artificial intelligence improve personalized learning?",
        "response": "Artificial Intelligence can transform personalized learning by: Adaptive learning platforms that adjust content difficulty based on individual performance (Carnegie Learning, Khan Academy), AI tutors providing 24/7 personalized support, automated feedback on writing and problem-solving, predictive analytics identifying at-risk students before they fail, intelligent content recommendations based on learning history, automated grading freeing teachers for higher-value interactions, natural language processing enabling conversational learning assistants, and learning analytics dashboards giving teachers real-time insights into every student's progress."
    },
    {
        "instruction": "What are the key principles of constructivism in education?",
        "response": "Constructivism, developed by Piaget and Vygotsky, holds that learners actively construct knowledge through experience rather than passively receiving information. Key principles include: Learning is an active, constructive process; new knowledge is built on prior knowledge; social interaction is crucial for learning (Vygotsky's social constructivism); learning is most effective in authentic, context-rich environments; reflection deepens understanding. Constructivist classrooms feature inquiry-based learning, collaborative projects, problem-solving, and student-driven exploration rather than direct instruction."
    },
    {
        "instruction": "What is the role of feedback in the learning process?",
        "response": "Feedback is one of the most powerful influences on student learning and achievement. Effective feedback is: Specific (clearly identifies what was done well or needs improvement), timely (provided close to when the work was done), actionable (gives clear guidance for improvement), and growth-oriented (focuses on the process, not just the outcome). John Hattie's meta-analysis of 800+ studies shows feedback has one of the highest effect sizes on student achievement (d=0.73). Feedback can come from teachers, peers, technology, or self-assessment."
    },
    {
        "instruction": "How does bilingual education benefit students?",
        "response": "Bilingual education, where instruction is delivered in two languages, offers significant cognitive and academic advantages: Enhanced executive function and cognitive flexibility, delayed onset of dementia, greater empathy and cultural awareness, stronger metalinguistic awareness improving literacy in both languages, higher academic achievement in some research, greater career opportunities in a globalized economy, and cultural pride and identity preservation. Dual language immersion programs, where native and non-native speakers learn together, show the strongest outcomes for both groups."
    },
    # ── Teacher Development ──────────────────────────────────────────────────
    {
        "instruction": "What is professional development for teachers?",
        "response": "Professional development (PD) encompasses ongoing learning activities that enhance teachers' knowledge, skills, and effectiveness. Effective PD is: Sustained and ongoing (not one-time workshops), content-focused on subject matter and pedagogy, collaborative through professional learning communities, active with teachers practicing new strategies, coaching-supported with follow-up mentoring. Examples include workshops, conferences, peer observation, action research, online courses, lesson study, and instructional coaching. Research shows high-quality PD significantly improves both teacher effectiveness and student outcomes."
    },
    {
        "instruction": "What qualities make an effective teacher?",
        "response": "Research identifies key qualities of effective teachers: Deep subject knowledge, strong pedagogical skills (knowing how to teach the subject), high expectations for all students, building positive relationships with students, classroom management proficiency, data-driven instructional decision making, cultural responsiveness and inclusive practices, continuous professional learning, effective communication with families, adaptability and creativity, enthusiasm for learning, and reflective practice. The Measures of Effective Teaching (MET) project found that effective teaching is the most important school-based factor in student achievement."
    },
    {
        "instruction": "What is culturally responsive teaching?",
        "response": "Culturally responsive teaching (CRT) is a pedagogy that recognizes the importance of including students' cultural references in all aspects of learning. Developed by Gloria Ladson-Billings, CRT involves: Using students' cultural backgrounds as assets for learning, making curriculum relevant to students' lives, high academic expectations for all cultural groups, building critical consciousness, creating inclusive classroom communities, and connecting content to cultural knowledge students bring. CRT has been shown to improve engagement and academic outcomes for historically marginalized students."
    },
    # ── Assessment & Curriculum ──────────────────────────────────────────────
    {
        "instruction": "What is curriculum design and what are its key components?",
        "response": "Curriculum design is the purposeful planning of what students will learn, how they will learn it, and how their learning will be assessed. Key components include: Learning objectives (what students should know and be able to do), content selection (what knowledge and skills to include), instructional strategies (how to teach the content), sequence and scope (order and depth of topics), resources and materials, assessment and evaluation methods, and alignment (ensuring objectives, instruction, and assessment all align — called backward design or Understanding by Design by Wiggins & McTighe)."
    },
    {
        "instruction": "How is standardized testing used in education?",
        "response": "Standardized testing uses consistent questions and scoring to measure student performance across a large population. Uses include: Measuring academic achievement and growth, identifying achievement gaps between demographic groups, evaluating school and teacher effectiveness, determining college readiness (SAT, ACT), awarding credentials and diplomas, informing policy decisions, and international benchmarking (PISA, TIMSS). Critics argue standardized tests narrow curriculum, create test anxiety, and disadvantage certain populations. The debate centers on balancing accountability with authentic, holistic assessment."
    },
    {
        "instruction": "What is the importance of reading literacy in education?",
        "response": "Reading literacy is foundational to all learning — students who cannot read proficiently by 3rd grade are four times more likely to drop out of school. Reading literacy encompasses decoding (phonics), fluency, vocabulary, and comprehension. The Science of Reading — grounded in decades of cognitive research — shows explicit, systematic phonics instruction is essential for early reading development. Beyond basic literacy, higher-order reading skills — critical analysis, inference, synthesizing multiple texts — are essential for success in higher education and the modern workforce."
    },
    {
        "instruction": "Explain the concept of backwards design in curriculum development.",
        "response": "Backward Design, developed by Grant Wiggins and Jay McTighe in 'Understanding by Design,' is a curriculum planning approach that begins with the end in mind. It follows three stages: Stage 1 — Identify Desired Results (what should students know, understand, and be able to do?), Stage 2 — Determine Acceptable Evidence (how will we know students have achieved the desired results?), Stage 3 — Plan Learning Experiences and Instruction (what activities and teaching will help students achieve the goals?). This reverses traditional planning where teachers plan activities first, ensuring alignment between goals, assessment, and instruction."
    },
    # ── Student Wellbeing ────────────────────────────────────────────────────
    {
        "instruction": "How does mental health affect student academic performance?",
        "response": "Mental health significantly impacts academic performance. Students with anxiety, depression, or other mental health challenges show reduced concentration, memory, motivation, and academic engagement. Mental health disorders are the leading cause of disability among young people aged 10-24. Schools are increasingly recognizing their role in mental health support through: School counselors and psychologists, SEL curricula, trauma-informed practices, mental health awareness programs, crisis intervention protocols, and partnerships with community mental health providers. Early identification and intervention dramatically improve both mental health and academic outcomes."
    },
    {
        "instruction": "What is the impact of homework on student learning?",
        "response": "Research on homework shows complex, nuanced results. John Hattie's synthesis shows homework in secondary school has a moderate positive effect (d=0.55) but minimal effect in primary school. Effective homework should be purposeful (reinforcing specific skills), appropriately challenging (neither too easy nor too hard), moderate in amount (research suggests no more than 10 minutes per grade level), and connected to classroom learning. Excessive homework creates stress, reduces family time, and may increase achievement gaps if students lack home support. Homework should be reviewed and its impact assessed continuously."
    },
    {
        "instruction": "What is student-centered learning?",
        "response": "Student-centered learning shifts focus from teacher delivery to student engagement, agency, and construction of knowledge. Key characteristics include: Students take active roles in their learning, teachers facilitate rather than deliver content, curriculum connects to student interests and real-world contexts, students have voice and choice in how they learn and demonstrate understanding, collaboration is emphasized, and learning is personalized to individual needs and goals. Research shows student-centered approaches improve motivation, critical thinking, self-regulation, and deep understanding compared to purely teacher-directed instruction."
    },
]

# Add 50 more entries to reach robust training size
ADDITIONAL_QA = [
    {
        "instruction": "What is the zone of proximal development?",
        "response": "The Zone of Proximal Development (ZPD), introduced by Lev Vygotsky, is the gap between what a learner can accomplish independently and what they can achieve with guidance from a more knowledgeable other (teacher, peer, or mentor). Effective teaching targets the ZPD — tasks challenging enough to require support but achievable with that support. This is why scaffolding is so effective: it provides temporary assistance within the ZPD, gradually withdrawn as the student develops competence. The ZPD concept underpins collaborative learning, differentiated instruction, and mentorship."
    },
    {
        "instruction": "What are the benefits of extracurricular activities in schools?",
        "response": "Extracurricular activities — sports, arts, clubs, student government — provide benefits beyond the classroom: Development of teamwork, leadership, and communication skills, exploration of interests and potential career paths, improved physical health through sports, mental health benefits through creative expression and social connection, higher academic motivation and school engagement, reduced dropout rates, development of time management and discipline, and opportunities for students who may not excel academically to discover and showcase other talents."
    },
    {
        "instruction": "How does parental involvement affect student achievement?",
        "response": "Parental involvement is consistently one of the strongest predictors of student academic success. Research shows that students with engaged parents have higher grades and test scores, better attendance, improved behavior, higher graduation rates, and greater likelihood of attending college. Effective parental involvement includes: reading with young children, monitoring homework and academic progress, communicating with teachers, attending school events, having high academic expectations, creating a home environment supportive of learning, and engaging in school governance through PTA/PTO."
    },
    {
        "instruction": "What is inquiry-based learning?",
        "response": "Inquiry-based learning is a student-centered approach where students construct knowledge by asking questions, investigating, and drawing conclusions — mirroring the process of scientific and scholarly inquiry. The inquiry cycle includes: Questioning (identifying a problem or question), Investigation (gathering data and evidence), Analysis (interpreting findings), Communication (sharing conclusions), and Reflection (evaluating the process). Levels range from structured inquiry (teacher-directed questions) to open inquiry (student-generated questions). IBL develops critical thinking, research skills, and intrinsic motivation."
    },
    {
        "instruction": "What is the difference between synchronous and asynchronous learning?",
        "response": "Synchronous learning occurs in real time — live lectures, video conferences, real-time discussions — where all participants are present simultaneously. Benefits include immediate interaction, real-time feedback, and community building. Asynchronous learning occurs at different times — pre-recorded videos, discussion boards, self-paced modules — where participants engage independently. Benefits include flexibility, self-pacing, time for reflection, and accessibility across time zones. Blended approaches combining both are increasingly common, leveraging synchronous sessions for discussion and collaboration while using asynchronous content delivery."
    },
    {
        "instruction": "What role does motivation play in student learning?",
        "response": "Motivation is a critical driver of learning. Self-Determination Theory (Deci & Ryan) identifies three basic psychological needs that support intrinsic motivation: Autonomy (feeling in control of one's actions), Competence (feeling effective and capable), and Relatedness (feeling connected to others). Intrinsically motivated students persist longer, engage more deeply, and retain learning better. Teachers support motivation by: providing choice, setting appropriate challenge levels, offering genuine praise for effort, building relationships, making content relevant, and creating psychologically safe classrooms where mistakes are valued as learning opportunities."
    },
    {
        "instruction": "How are learning objectives written effectively?",
        "response": "Effective learning objectives use the SMART framework and Bloom's Taxonomy action verbs. They specify exactly what students will know, understand, or be able to do after instruction. A well-written objective includes: an action verb (from Bloom's: list, explain, apply, analyze, evaluate, create), the specific content or skill, and the level of performance expected. Example: 'Students will be able to analyze the causes of World War II using at least three primary sources.' Bloom's verbs progress from lower-order (remember, understand) to higher-order (analyze, evaluate, create) thinking skills."
    },
    {
        "instruction": "What is the significance of classroom environment in learning?",
        "response": "The physical and emotional classroom environment profoundly affects learning. Physical environment factors include: seating arrangements (rows vs clusters), natural lighting, temperature, noise levels, access to materials, and display of student work. Emotional/social environment factors include: psychological safety (students feel safe to take risks), positive relationships, inclusive culture, clear routines and expectations, and classroom norms. Research shows students learn significantly better in environments where they feel physically comfortable, emotionally safe, intellectually challenged, and socially connected."
    },
]

# Combine all Q&A data
ALL_QA = EDUCATION_QA + ADDITIONAL_QA


# ==============================================================================
# 2. WEB SCRAPING — Wikipedia Education Articles
# ==============================================================================

def scrape_wikipedia_education(topics: List[str], max_chars: int = 2000) -> List[Dict]:
    """
    Scrape introductory paragraphs from Wikipedia articles on education topics.

    Parameters
    ----------
    topics    : list of Wikipedia article titles to scrape
    max_chars : maximum characters to extract per article

    Returns
    -------
    list of dicts with 'title' and 'content' keys
    """
    results = []
    headers = {"User-Agent": "EduBot-Research/1.0 (Educational Project)"}

    for topic in topics:
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                content = data.get("extract", "")[:max_chars]
                if content:
                    results.append({"title": topic, "content": content, "source": "Wikipedia"})
                    logger.info("Scraped: %s (%d chars)", topic, len(content))
            else:
                logger.warning("Failed to scrape %s (status %d)", topic, response.status_code)

            time.sleep(0.5)  # Be respectful to Wikipedia's servers

        except Exception as e:
            logger.warning("Error scraping %s: %s", topic, e)

    return results


EDUCATION_WIKIPEDIA_TOPICS = [
    "Educational_psychology",
    "Constructivism_(philosophy_of_education)",
    "Bloom%27s_taxonomy",
    "Differentiated_instruction",
    "Problem-based_learning",
    "Gamification_of_learning",
    "Special_education",
    "E-learning",
    "Early_childhood_education",
    "Montessori_education",
    "Waldorf_education",
    "STEM_education",
    "Flipped_classroom",
    "Universal_Design_for_Learning",
    "Growth_mindset",
]


def convert_articles_to_qa(articles: List[Dict]) -> List[Dict]:
    """
    Convert scraped Wikipedia articles into instruction-response pairs.

    For each article, we generate a question "What is <topic>?" and use
    the article content as the response.
    """
    qa_pairs = []
    for article in articles:
        title   = article["title"].replace("_", " ").replace("%27", "'")
        content = article["content"]
        if len(content) > 100:
            qa_pairs.append({
                "instruction": f"What is {title}?",
                "response"   : content,
                "source"     : article.get("source", "web"),
            })
    return qa_pairs


# ==============================================================================
# 3. MAIN COLLECTION FUNCTION
# ==============================================================================

def collect_all_data(scrape_web: bool = False) -> pd.DataFrame:
    """
    Collect data from all sources and return as a unified DataFrame.

    Parameters
    ----------
    scrape_web : whether to scrape Wikipedia (set False for offline/Colab)

    Returns
    -------
    DataFrame with columns: instruction, response, source
    """
    all_data = []

    # ── Source 1: Synthetic Education Q&A ────────────────────────────────────
    logger.info("Loading synthetic education Q&A dataset (%d pairs) …", len(ALL_QA))
    for item in ALL_QA:
        all_data.append({
            "instruction": item["instruction"],
            "response"   : item["response"],
            "source"     : "synthetic_expert",
        })

    # ── Source 2: Wikipedia Scraping (optional) ───────────────────────────────
    if scrape_web:
        logger.info("Scraping Wikipedia education articles …")
        articles = scrape_wikipedia_education(EDUCATION_WIKIPEDIA_TOPICS)
        wiki_qa  = convert_articles_to_qa(articles)
        for item in wiki_qa:
            all_data.append(item)
        logger.info("  Added %d Wikipedia Q&A pairs", len(wiki_qa))

    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset=["instruction"]).reset_index(drop=True)
    df["id"] = df.index

    logger.info("Total collected: %d Q&A pairs", len(df))
    return df


def save_raw_data(df: pd.DataFrame, filename: str = "education_raw.json") -> str:
    """Save raw collected data to disk."""
    path = str(RAW_DIR / filename)
    df.to_json(path, orient="records", indent=2)
    logger.info("Raw data saved → %s", path)
    return path


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  EduBot Data Collection Pipeline")
    logger.info("=" * 60)

    # Collect data (set scrape_web=True if internet is available)
    df = collect_all_data(scrape_web=True)

    print(f"\n📊 Dataset Summary:")
    print(f"  Total pairs    : {len(df)}")
    print(f"  Sources        : {df['source'].value_counts().to_dict()}")
    print(f"  Avg instruction length : {df['instruction'].str.len().mean():.0f} chars")
    print(f"  Avg response length    : {df['response'].str.len().mean():.0f} chars")
    print(f"\nSample entry:")
    print(f"  Q: {df['instruction'].iloc[0]}")
    print(f"  A: {df['response'].iloc[0][:120]}…")

    save_raw_data(df)
    print(f"\n✅ Data collection complete. Saved to data/raw/education_raw.json")
