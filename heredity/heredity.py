import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def probability_inheritence(num_genes_of_parent, is_inherited):
    """
    Given the number of copies of the gene the parent has (0, 1 or 2), 
    and the inheritence outcome (True if child inherits gene from this parent, else False),
    computes the probability of the inheritance outcome for this parent.
    """
    # If parent has no copy of gene, can only pass it on via mutation
    if num_genes_of_parent == 0:
        if is_inherited:
            return PROBS["mutation"]
        else:
            return 1 - PROBS["mutation"]

    # If parent has 1 copy of gene, 50/50 chance of whether they pass it on
    elif num_genes_of_parent == 1:
        return 0.5
    
    # If parent has 2 copies of gene, only way they won't pass it on is via mutation
    elif num_genes_of_parent == 2:
        if is_inherited:
            return 1 - PROBS["mutation"]
        else:
            return PROBS["mutation"]
    
    else:
        raise Exception("invalid input")


def num_genes_of_person(person, one_gene, two_genes):
    if person in one_gene:
        return 1
    elif person in two_genes:
        return 2
    else:
        return 0
    

def joint_probability(people, one_gene, two_genes, have_trait):
    probability = 1

    for person in people:
        num_genes = num_genes_of_person(person, one_gene, two_genes)
        has_trait = person in have_trait
        if people[person]['mother'] is None and people[person]['father'] is None:
            probability *= PROBS["gene"][num_genes] * PROBS["trait"][num_genes][has_trait]
        else:
            num_genes_mother = num_genes_of_person(people[person]['mother'], one_gene, two_genes)
            num_genes_father = num_genes_of_person(people[person]['father'], one_gene, two_genes)
            if num_genes == 0:
                probability *= probability_inheritence(num_genes_mother, False) * probability_inheritence(num_genes_father, False)
            elif num_genes == 1:
                probability *= probability_inheritence(num_genes_mother, True) * probability_inheritence(num_genes_father, False) \
                                + probability_inheritence(num_genes_mother, False) * probability_inheritence(num_genes_father, True)
            elif num_genes == 2:
                probability *= probability_inheritence(num_genes_mother, True) * probability_inheritence(num_genes_father, True)
            probability *= PROBS["trait"][num_genes][has_trait]

    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    for person in probabilities:
        genes = num_genes_of_person(person, one_gene, two_genes)

        trait = person in have_trait

        probabilities[person]["gene"][genes] += p
        probabilities[person]["trait"][trait] += p


def normalize(probabilities):
    for person in probabilities:
        trait_sum = sum(probabilities[person]["trait"].values())
        gene_sum = sum(probabilities[person]["gene"].values())

        for gene in probabilities[person]["gene"]:
            probabilities[person]["gene"][gene] /= gene_sum

        for trait in probabilities[person]["trait"]:
            probabilities[person]["trait"][trait] /= trait_sum


if __name__ == "__main__":
    main()
