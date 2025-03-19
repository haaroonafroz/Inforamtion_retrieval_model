import os
import pandas as pd
import argparse
import random
from pathlib import Path

def create_sample_dataset(target_dir, dataset_name, num_documents=20, num_questions=10):
    """
    Create a sample dataset for RAG evaluation.
    
    Args:
        target_dir: Directory to store the dataset
        dataset_name: Name of the dataset (wikitext, chatlogs, state_of_the_union)
        num_documents: Number of documents to create
        num_questions: Number of questions to create
    """
    print(f"Creating sample {dataset_name} dataset...")
    
    # Create directories
    corpora_dir = os.path.join(target_dir, 'corpora')
    os.makedirs(corpora_dir, exist_ok=True)
    
    # Sample text for different datasets
    if dataset_name == "wikitext":
        texts = [
            f"Wikipedia is a free online encyclopedia created and edited by volunteers around the world. "
            f"It was launched on January 15, 2001, by Jimmy Wales and Larry Sanger. "
            f"Wikipedia consists of more than 40 million articles in more than 300 languages. "
            f"The English Wikipedia alone has over 6 million articles. "
            f"Wikipedia is the largest and most popular general reference work on the Internet.",
            
            f"Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence "
            f"of humans and other animals. AI research defines artificial intelligence as a machine that can learn "
            f"from experience, adjust to new inputs, and perform human-like tasks. "
            f"Modern AI techniques include machine learning, deep learning, and natural language processing.",
            
            f"The history of computing hardware starting with the earliest mechanical calculating devices up to the "
            f"modern day. Before the 20th century, most calculations were done by humans. Early mechanical tools to "
            f"help humans with calculations included the abacus and the slide rule. "
            f"The first electronic computers were developed in the mid-20th century.",
            
            f"The World Wide Web (WWW), commonly known as the Web, is an information system where documents and "
            f"other web resources are identified by Uniform Resource Locators (URLs), which may be interlinked by "
            f"hyperlinks. Web resources are accessible through the Internet. The Web was invented by Tim Berners-Lee "
            f"at CERN in 1989.",
            
            f"Machine learning (ML) is a field of study in artificial intelligence concerned with the development "
            f"and study of algorithms that can learn from and make decisions based on data. "
            f"Machine learning algorithms are used in a wide variety of applications, "
            f"such as in medicine, email filtering, speech recognition, and computer vision.",
        ]
    elif dataset_name == "chatlogs":
        texts = [
            f"User: How can I reset my password?\nSupport: To reset your password, click on the 'Forgot Password' link "
            f"on the login page. You will receive an email with instructions to reset your password. "
            f"If you don't receive the email, please check your spam folder.",
            
            f"User: My account is showing incorrect balance.\nSupport: I'm sorry to hear about the balance issue. "
            f"Please provide your account number and the last transaction you made. "
            f"We'll investigate the issue and get back to you within 24 hours.",
            
            f"User: How do I upgrade my subscription?\nSupport: To upgrade your subscription, go to your account "
            f"settings and click on 'Subscription'. You'll see an 'Upgrade' button next to your current plan. "
            f"Click on it and follow the instructions to complete the upgrade.",
            
            f"User: The app keeps crashing on my phone.\nSupport: I'm sorry to hear that. Could you please tell me "
            f"what type of phone you're using and the app version? Also, have you tried restarting your phone? "
            f"That often resolves temporary issues.",
            
            f"User: I can't download the latest update.\nSupport: Let's troubleshoot this issue. First, check your "
            f"internet connection. Then, make sure you have enough storage space on your device. "
            f"If the problem persists, try uninstalling and reinstalling the app.",
        ]
    else:  # state_of_the_union
        texts = [
            f"Mr. Speaker, Madam Vice President, our First Lady and Second Gentleman. "
            f"Members of Congress and the Cabinet. Leaders of our military. Mr. Chief Justice, "
            f"Associate Justices, and retired Justices of the Supreme Court. "
            f"Distinguished guests, my fellow Americans.",
            
            f"Tonight I want to talk about what we've done, what we have to do, and the extraordinary "
            f"opportunity that lies before us. We meet tonight as Americans, and we have each been "
            f"given a great gift - the freedom to determine our destiny and the obligation to "
            f"fulfill our common purpose.",
            
            f"In our first year, we cut the deficit by $350 billion. In our second year, we cut it by "
            f"more than $900 billion. And we've continued to reduce the deficit while growing "
            f"our economy and creating a record number of jobs. In fact, we've created over 500,000 "
            f"manufacturing jobs since I took office.",
            
            f"As we gather here tonight, we're writing the next chapter in the great American story - "
            f"a story of progress and resilience unlike any in the world. "
            f"When world leaders ask me to define America - and they do, believe me - I define our country "
            f"in one word: possibilities.",
            
            f"For decades, the middle class has been hollowed out. Too many jobs moved overseas. "
            f"Factories closed down. Once-thriving cities and towns became shadows of what they used to be. "
            f"The pandemic only made things worse. But now America is coming back.",
        ]
    
    # Create corpus documents
    for i in range(num_documents):
        # Select random text and expand it with repetitions to make it longer
        base_text = random.choice(texts)
        expanded_text = base_text
        
        # Repeat text to make document longer (with some variations)
        for _ in range(random.randint(5, 10)):
            expanded_text += "\n\n" + random.choice(texts)
        
        # Write to file
        with open(os.path.join(corpora_dir, f"{dataset_name}_{i+1}.txt"), 'w', encoding='utf-8') as f:
            f.write(expanded_text)
    
    print(f"Created {num_documents} corpus documents for {dataset_name}")
    
    # Create questions and answers
    questions = []
    answers = []
    dataset_labels = []
    
    if dataset_name == "wikitext":
        qa_pairs = [
            ("When was Wikipedia launched?", "Wikipedia was launched on January 15, 2001, by Jimmy Wales and Larry Sanger."),
            ("What is artificial intelligence?", "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence of humans and other animals."),
            ("Who invented the World Wide Web?", "The Web was invented by Tim Berners-Lee at CERN in 1989."),
            ("What is machine learning used for?", "Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision."),
            ("How many articles does English Wikipedia have?", "The English Wikipedia has over 6 million articles."),
        ]
    elif dataset_name == "chatlogs":
        qa_pairs = [
            ("How do I reset my password?", "To reset your password, click on the 'Forgot Password' link on the login page. You will receive an email with instructions to reset your password."),
            ("What should I do if my account shows incorrect balance?", "Provide your account number and the last transaction you made. The support team will investigate the issue."),
            ("How can I upgrade my subscription?", "Go to your account settings and click on 'Subscription'. You'll see an 'Upgrade' button next to your current plan."),
            ("What should I do if the app keeps crashing?", "Restart your phone, check the app version, and make sure your operating system is up to date."),
            ("How do I fix download issues for updates?", "Check your internet connection and make sure you have enough storage space on your device. If the problem persists, try uninstalling and reinstalling the app."),
        ]
    else:  # state_of_the_union
        qa_pairs = [
            ("How much did the administration cut the deficit in the second year?", "In the second year, they cut the deficit by more than $900 billion."),
            ("How many manufacturing jobs were created since the administration took office?", "Over 500,000 manufacturing jobs were created since the administration took office."),
            ("What word does the speaker use to define America to world leaders?", "The speaker defines America in one word: possibilities."),
            ("Who does the speaker address at the beginning of the speech?", "The speaker addresses Mr. Speaker, Madam Vice President, the First Lady and Second Gentleman, Members of Congress and the Cabinet, and others."),
            ("What issue does the speaker mention about the middle class?", "For decades, the middle class has been hollowed out, with too many jobs moved overseas and factories closed down."),
        ]
    
    # Generate required number of questions
    for i in range(num_questions):
        # Use predefined QA pairs or generate random ones if needed
        if i < len(qa_pairs):
            question, answer = qa_pairs[i]
        else:
            idx = random.randint(0, len(qa_pairs) - 1)
            question, answer = qa_pairs[idx]
            # Add some variation to avoid exact duplicates
            question = question + f" (variation {i})"
        
        questions.append(question)
        answers.append(answer)
        dataset_labels.append(dataset_name)
    
    # Create questions dataframe
    questions_df = pd.DataFrame({
        'dataset': dataset_labels,
        'question': questions,
        'golden_text': answers
    })
    
    return questions_df

def main():
    parser = argparse.ArgumentParser(description="Create sample datasets for RAG evaluation")
    parser.add_argument("--target_dir", type=str, default="data", 
                        help="Directory to store the datasets")
    parser.add_argument("--datasets", type=str, nargs="+", 
                        default=["wikitext", "chatlogs", "state_of_the_union"],
                        help="List of datasets to create")
    
    args = parser.parse_args()
    
    # Create target directory if it doesn't exist
    os.makedirs(args.target_dir, exist_ok=True)
    
    # Create sample datasets
    all_questions = []
    for dataset_name in args.datasets:
        questions_df = create_sample_dataset(args.target_dir, dataset_name)
        all_questions.append(questions_df)
    
    # Combine all questions and save to CSV
    combined_df = pd.concat(all_questions, ignore_index=True)
    questions_path = os.path.join(args.target_dir, 'questions_df.csv')
    combined_df.to_csv(questions_path, index=False)
    
    print(f"\nSample datasets created successfully!")
    print(f"Corpus documents are in: {os.path.join(args.target_dir, 'corpora')}")
    print(f"Questions file saved to: {questions_path}")

if __name__ == "__main__":
    main() 