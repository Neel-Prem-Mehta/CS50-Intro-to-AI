import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = dict()

    for pages in corpus:
        if pages in corpus[page]:
            distribution[pages] = damping_factor/len(corpus[page]) + (1 - damping_factor)/len(corpus)
        else:
            distribution[pages] = (1 - damping_factor)/len(corpus)
    
    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = []
    visited = dict()
    for page in corpus:
        pages.append(page)
        visited[page] = 0

    page = pages[random.randrange(0, len(pages))]
    visited[page] += 1
    n = n-1

    for x in range(n):
        total_prob = 0

        future = transition_model(corpus, page, damping_factor)
        rand_value = random.random()

        for pg in future:
            total_prob += future[pg]
            if rand_value < total_prob:
                page = pg
                break
        
        visited[page] += 1

    for pg in visited:
        visited[pg] = visited[pg]/(n+1)
    
    return visited


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    prob = dict()
    
    for page in corpus:
        prob[page] = 1/len(corpus)

    biggest_change = 1

    while biggest_change >= 0.001:
        for page in corpus:
            l = corpus[page]
            pre_update = prob[page]
            prob[page] = (1 - damping_factor)/len(corpus)
            for pages in corpus:
                links = corpus[pages]
                if page in links:
                    prob[page] += prob[pages]/len(links)
            change = pre_update - prob[page]
            if abs(change) > biggest_change:
                biggest_change = abs(change)
    
    return prob


if __name__ == "__main__":
    main()
