import setuptools

REPO_NAME = "Rag_Based_Chatbot_Project"
AUTHOR_USER_NAME = "Akhilpm156"
AUTHOR_EMAIL = "akhilpm156@gmail.com"


setuptools.setup(
    name='rag_project',
    version= "0.0.0",
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for Rag Chatbot Project",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
    )