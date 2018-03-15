""" Functions that run SPARQL querys
e.g. search dbpedia for the birthdates of authors

"""
from SPARQLWrapper import SPARQLWrapper, JSON, XML, N3, RDF
import json

# An example query
query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbpedia: <http://dbpedia.org/resource/>
PREFIX ontology: <http://dbpedia.org/ontology/>

SELECT distinct ?s  ?author
WHERE {
?s rdf:type ontology:Book;
  ontology:author ?author .
}
LIMIT 3
"""


def search(query):
    # results :: dict
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results


def book_info(title='The_Wonderful_Wizard_of_Oz'):
    query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbp: <http://dbpedia.org/resource/>
PREFIX dbo: <http://dbpedia.org/ontology/>

SELECT ?book ?title ?author ?genre ?date ?birthDate
WHERE {
    ?book rdf:type dbo:Book;
        dbo:author ?author.
    FILTER regex(str(?book), "%s").
    OPTIONAL { ?book dbo:releaseDate ?date. }
    OPTIONAL { ?book dbo:literaryGenre ?genre. }
    OPTIONAL { ?author dbo:birthDate ?birthDate. }
}
LIMIT 3
""" % title
    return search(query)


def author_info(name='L._Frank_Baum'):
    query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbpedia: <http://dbpedia.org/resource/>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbpedia-owl: <http://dbpedia.org/ontology/>

SELECT ?author ?date
WHERE {
    ?author dbo:child ?child
    FILTER regex(str(?child), "%s") .
    ?author dbo:birthDate ?date .
}
LIMIT 10
""" % name
    return search(query)
