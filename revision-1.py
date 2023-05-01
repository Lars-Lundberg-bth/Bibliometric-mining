import pybliometrics
import copy

class Dictentry:
    def __init__(self):
        self.kw_set = set() # create empty set
        self.age_counter = 0
        self.citation_count = 0
    def add_to_set(self, new_eid, nr_refs = 0):
        if not new_eid in self.kw_set:
            self.kw_set.add(new_eid)
            self.citation_count += nr_refs
    def clear_set(self):
        self.kw_set = set()
    def get_count(self):
        return len(self.kw_set)
    def clear_set(self):
        self.kw_set = set() # set existing set to empty set
    def make_union(self, added_set):
        self.kw_set = self.kw_set.union(added_set)
    def get_set(self):
        return self.kw_set
    def add_age_counter(self,value): # age_counter is used for many purposes, not only to handle the age_factor
        self.age_counter += value
    def get_age_counter(self):
        return self.age_counter
    def get_citations(self):
        return self.citation_count

age_factor = 1.5
max_key_word_length = 5
minimum_kw_occaision = 20

years = (2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022)
first_part_of_search_str = 'TITLE-ABS-KEY ( {big data} ) AND (PUBYEAR = '
# 'TITLE-ABS-KEY ( {big data} ) AND ("history" OR "archaeology" OR "languages" OR "literature" OR "philosophy" OR "ethics" OR "religion" OR "arts" OR "music") AND (PUBYEAR = '
# 'TITLE-ABS-KEY ( {big data} ) AND ("psychology" OR "economics" OR "business" OR "educational science" OR "sociology" OR "law" OR "political science" OR "social and economic geography" OR "media and communications") AND (PUBYEAR = '
# 'TITLE-ABS-KEY ( {big data} ) AND ("agriculture" OR "forestry" OR "fishery" OR "animal" OR "dairy" OR "veterinary" OR "agriculture biotechnology") AND (PUBYEAR = '
# 'TITLE-ABS-KEY ( {big data} ) AND ("civil engineering" OR "electrical engineering" OR "electronic engineering" OR "information engineering" OR "mechanical engineering" OR "chemical engineering" OR "materials engineering" OR "medical engineering" OR "environmental engineering" OR "environmental biotechnology" OR "industrial biotechnology" OR "nano-technology" OR "nano technology") AND (PUBYEAR = '
# 'TITLE-ABS-KEY ( {big data} ) AND ("medicine" OR "health" OR "health biotechnology") AND (PUBYEAR = '
# 'TITLE-ABS-KEY ( {big data} ) AND ("mathemathics" OR "computer end information" OR "physics" OR "chemistry" OR "environmental science" OR "biology") AND (PUBYEAR = '

black_list = ['research','data','big data','big data technology','big data management','it', 'its','ict','analysis','information','system','processing','computing','analytics','challenges','model','models','technology',
'development','methods','technologies', 'learning','framework','algorithm','algorithms','process','future','time','management','intelligence','design','knowledge','service',
'intelligent','digital','quality','accuracy','environment','platform','prediction','industry','services','science','application','applications','internet','web','software',
'business','resources','value','evaluation','impact','control','smart','detection','cost','volume','context',
'data analytics','data analysis','big data analysis','big data analytics', 'the era of big data','data processing','information technology','big data era',
'construction','network','networks','data sets','data set','data collection', 'data management',
'big data processing','large data','data sources','large scale','bigdata','big-data','big data applications','big data technologies','massive data','modeling','metadata',
'big data analytics capability','big data analytics capabilities','monitoring','performance','efficiency','optimization','methodology','benchmark','classification','clustering',
'semantics','measurement'] 

black_set = set()
for black_kw in black_list:
    black_set.add(tuple(black_kw.split()))
black_set = frozenset(black_set)

syn_list = [
    [['iot', 'internet of things', 'the internet of things', 'internet of things iot','internet-of-things','industrial internet of things'],['Internet of things']],
    [['cloud','cloud computing','cloud services','cloud platform','cloud storage'],['Cloud computing']],
    [['parallel','distributed','gpu','cuda','parallel processing','parallel algorithms','parallel algorithm','distributed processing','parallel computing','distributed computing','load balancing','distributed systems','distributed system','high performance computing','high-performance computing','hpc'],['Parallel and distributed computing']],
    [['data mining','mining','educational data mining','text mining','big data mining','data science','process mining'],['Data mining']],
    [['machine learning','machine learning algorithms'],['Machine learning']],
    [['artificial intelligence','ai','artificial intelligence ai'],['Artificial intelligence']],
    [['storage','database','databases','data lake','nosql','data storage','mongodb','file system','database systems','hbase','cassandra'],['Storage']],
    [['social media','social big data','social network','social networks','social network analysis'],['Social media']],
    [['teaching','education','higher education','e-learning'],['Teaching and education']],
    [['health','healthcare','medical informatics','mhealth','electronic health records','health care','public health','digital health','telemedicine','personalized medicine','precision medicine','health informatics','medical big data','epidemiology','medical imaging','e-health','diabetes','breast cancer','genetics'],['Healthcare']],
    [['security','authentication','data security','data protection','privacy','cloud security','privacy preservation','privacy preserving','privacy-preserving','data privacy','differential privacy','big data security','intrusion detection','anomaly detection','encryption','homomorphic encryption','cryptography','cyber security','trust','information security','cybersecurity','privacy protection','network security'],['Security and privacy']], 
    [['deep learning','deep neural network','deep neural networks','deep learning dl','deep reinforcement learning','neural network','neural networks','artificial neural networks','artificial neural network','convolutional neural networks','convolutional neural network','convolution neural network','cnn','recurrent neural network','recurrent neural network','recurrent neural networks','bp neural network'],['DL and neural networks']],
    [['industry 4 0', 'manufacturing','smart manufacturing','digital twin','smart factory','automation','robotics','industrial big data','supply chain management','supply chain','predictive maintenance'],['Manufacturing']],
    [['edge computing','fog computing'],['Edge and fog computing']],
    [['smart city','urban computing','smart cities','citizen science','smart home'],['Smart cities']], 
    [['review','literature review','survey','systematic literature review','systematic review','bibliometrics','bibliometric analysis'],['Review']],
    '''
    [['mathemathics', 'computer end information', 'physics', 'chemistry', 'environmental science', 'biology'],['Natural sciences']],
    [['civil engineering', 'electrical engineering', 'electronic engineering', 'information engineering', 'mechanical engineering', 'chemical engineering', 'materials engineering', 'medical engineering', 'environmental engineering', 'environmental biotechnology', 'industrial biotechnology', 'nano-technology', 'nano technology'],['Engineering and technology']],
    [['medicine', 'health', 'health biotechnology'],['Medical and health sciences']],
    [[''],['Agriculture sciences']]
    '''
    ]
    
syn_dict = {}
for synonym in syn_list:
    print(synonym[1][0])
    composed_kw = tuple(synonym[1][0].split())
    print('Composed kw: ', composed_kw)
    for old_kw in synonym[0]:
        syn_dict[tuple(old_kw.split())] = composed_kw
        print('Old kw: ', tuple(old_kw.split()))

def insert_dict(eid, kw_tup, dict, create_flag = False, nr_refs = 0): 
    if not (kw_tup in black_set): 
        if kw_tup in syn_dict:
            kw_tup = syn_dict[kw_tup]
        if kw_tup in dict:
            if not eid in dict[kw_tup].get_set():
                dict[kw_tup].add_to_set(eid, nr_refs)
        elif create_flag:
            dict[kw_tup] = Dictentry() # create a new entry 
            dict[kw_tup].add_to_set(eid)    

def clean_text(txt):
    sentence_splitters = ".!?"
    remove_characters = "',:;()[]" + '"' + "{" + "}"
    txt = txt.lower()
    for character in remove_characters + sentence_splitters:
        txt = txt.replace(character,' ')
    return txt

def increment_akw(eid, txt, dict, create_flag = False, nr_refs = 0):
    txt = clean_text(txt)
    sentence_list = txt.split('|') 
    for kw in sentence_list:
        insert_dict(eid,tuple(kw.split()) ,dict, create_flag, nr_refs = nr_refs)
    return len(sentence_list)

def increment_title_and_abstract(eid, txt, dict, is_title = True, nr_refs = 0):
    txt = txt = clean_text(txt)
    sentence_list = txt.split() 
    for kw_length in range(max_key_word_length):
        for position in range(len(sentence_list) - kw_length):
            if is_title or (kw_length > 0):
                insert_dict(eid,tuple(sentence_list[position:(position+kw_length+1)]), dict, nr_refs = nr_refs)
            
my_dictionary = {}

keyw_count = 0
no_keyw = 0
no_of_articles = 0

pybliometrics.scopus.utils.create_config(["af6e11a84869dc42fe24d29d18c95c53"])

# First pass: finding all key words

year_list_citations = []

for year in years: 
    search_str = first_part_of_search_str + str(year) + ')'
    print(search_str)
    s = pybliometrics.scopus.ScopusSearch(search_str, refresh = False)
    print("number of articles:", len(s.results))
    no_of_articles += len(s.results)
    no_of_citations = 0 
    documents_with_aff_country_this_year = 0
    for doc in s.results:
        if doc.authkeywords:
            keyw_count += 1
            # print("number of citations: ", doc.citedby_count)
            no_keyw += increment_akw(doc.eid, doc.authkeywords, my_dictionary, create_flag = True)
        if doc.affiliation_country:
                no_of_citations += doc.citedby_count
                documents_with_aff_country_this_year += 1
    # year_list_citations += [(year,no_of_citations,len(s.results),no_of_citations/len(s.results))]
    year_list_citations += [(year,no_of_citations,documents_with_aff_country_this_year,no_of_citations/documents_with_aff_country_this_year)]

print("Year list citation")
print(year_list_citations)

print("Number of documents:", no_of_articles)
print("Documents with keywords: ", keyw_count)
print("Number of keywords: ", no_keyw)
print("Number of unique keywords: ", len(my_dictionary))
print("Maximum key word length: ", max_key_word_length)

# Second pass: counting number of documents that mention each key word

sort_list = list(my_dictionary.items())
sort_list.sort(key = lambda x: x[1].get_count(), reverse = True)
i = 0
active_kws = 0
my_new_dictionary = {}
print("len(my_dictionary)", len(my_dictionary))
while sort_list[i][1].get_count() > minimum_kw_occaision: # only use key words that are present in 20 or more documents
    if (len(sort_list[i][0]) > 1) or (sort_list[i][1].get_count() > 4*minimum_kw_occaision):
        my_new_dictionary[sort_list[i][0]] = Dictentry()
        active_kws += 1
    i += 1 
my_dictionary = my_new_dictionary

year_list = []
year_list_countries = []

top_list = [('Machine', 'learning'), ('DL', 'and', 'neural', 'networks'), ('Internet', 'of', 'things'), ('Data', 'mining'), ('Cloud', 'computing'), ('Artificial', 'intelligence'), ('Healthcare',),('Security', 'and', 'privacy'), ('Review',),('Manufacturing',)]
# top_list = [('Machine', 'learning'), ('Internet', 'of', 'things'), ('Data', 'mining'), ('Cloud', 'computing'), ('Artificial', 'intelligence'), ('Healthcare',),('Deep', 'learning'),('Security', 'and', 'privacy'),('Review',),('Neural', 'networks'), ('Manufacturing',) ]

china_count = 0
china_accumulated = 0
china_total_citations = 0
china_kw_citation_list = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
us_count = 0
us_accumulated = 0
us_total_citations = 0
us_kw_citation_list = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
eu_count = 0
eu_accumulated = 0
eu_total_citations = 0
eu_kw_citation_list = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
other_count = 0
other_accumulated = 0
other_total_citations = 0
other_kw_citation_list = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

all_accumulated = 0
all_count = 0

number_of_docs_witout_aff_country = 0

eu_countries = {"austria", "belgium", "bulgaria", "croatia", "republic-of-cyprus", "czech-republic", "denmark", "estonia", "finland", "france", "germany", "greece", "hungary", "ireland", "italy", "latvia", "lithuania", "luxembourg", "malta", "netherlands", "poland", "portugal", "romania", "slovakia", "slovenia", "spain", "sweden"}

old_year_list_citations = copy.deepcopy(year_list_citations)

for year in years:
    search_str = first_part_of_search_str + str(year) + ')'
    print(search_str)
    year_dictionary = copy.deepcopy(my_dictionary)
    citation_tup = year_list_citations.pop(0)

    print("Citation tuple:",citation_tup)

    s = pybliometrics.scopus.ScopusSearch(search_str, refresh = False)
    print("number of articles:", len(s.results))

    for doc in s.results:
        # Check title
        if doc.title:
            increment_title_and_abstract(doc.eid, doc.title, year_dictionary, nr_refs = doc.citedby_count)

        # Check key words
        if doc.authkeywords:
            increment_akw(doc.eid, doc.authkeywords, year_dictionary, nr_refs = doc.citedby_count)


        # Check abstract
        if doc.description:
            increment_title_and_abstract(doc.eid, doc.description, year_dictionary, is_title = False, nr_refs = doc.citedby_count)

        if doc.affiliation_country:
            all_accumulated += doc.citedby_count/citation_tup[3]
            all_count += 1
            aff_country = doc.affiliation_country
            aff_country = aff_country.replace(' ','-')
            countries = clean_text(aff_country)
            country_list = countries.split() 
            number_of_countries = len(country_list)
            for country in country_list:
                if country != 'none':
                    if country == 'china':
                        china_total_citations += doc.citedby_count/number_of_countries
                        china_accumulated += doc.citedby_count/citation_tup[3]/number_of_countries
                        china_count += 1/number_of_countries
                        for key_word in top_list:
                            if doc.eid in year_dictionary[key_word].get_set():
                                china_kw_citation_list[top_list.index(key_word)][0] += doc.citedby_count/number_of_countries
                                china_kw_citation_list[top_list.index(key_word)][1] += doc.citedby_count/citation_tup[3]/number_of_countries
                                china_kw_citation_list[top_list.index(key_word)][2] += 1/number_of_countries
                    elif (country == 'united-states') or (country == 'canada'):
                        us_total_citations += doc.citedby_count/number_of_countries
                        us_accumulated += doc.citedby_count/citation_tup[3]/number_of_countries
                        us_count += 1/number_of_countries
                        for key_word in top_list:
                            if doc.eid in year_dictionary[key_word].get_set():
                                us_kw_citation_list[top_list.index(key_word)][0] += doc.citedby_count/number_of_countries
                                us_kw_citation_list[top_list.index(key_word)][1] += doc.citedby_count/citation_tup[3]/number_of_countries
                                us_kw_citation_list[top_list.index(key_word)][2] += 1/number_of_countries
                    elif country in eu_countries or (year < 2020 and country == "united-kingdom"):
                        eu_total_citations += doc.citedby_count/number_of_countries
                        eu_accumulated += doc.citedby_count/citation_tup[3]/number_of_countries
                        eu_count += 1/number_of_countries
                        for key_word in top_list:
                            if doc.eid in year_dictionary[key_word].get_set():
                                eu_kw_citation_list[top_list.index(key_word)][0] += doc.citedby_count/number_of_countries
                                eu_kw_citation_list[top_list.index(key_word)][1] += doc.citedby_count/citation_tup[3]/number_of_countries
                                eu_kw_citation_list[top_list.index(key_word)][2] += 1/number_of_countries
                    else:
                        other_total_citations += doc.citedby_count/number_of_countries
                        other_accumulated += doc.citedby_count/citation_tup[3]/number_of_countries
                        other_count += 1/number_of_countries
                        for key_word in top_list:
                            if doc.eid in year_dictionary[key_word].get_set():
                                other_kw_citation_list[top_list.index(key_word)][0] += doc.citedby_count/number_of_countries
                                other_kw_citation_list[top_list.index(key_word)][1] += doc.citedby_count/citation_tup[3]/number_of_countries
                                other_kw_citation_list[top_list.index(key_word)][2] += 1/number_of_countries
                else:
                    print("COUNTRY IS NONE", aff_country)
        else:
            number_of_docs_witout_aff_country += 1
    year_list += [(year,year_dictionary)]
    year_list_countries += [(year,china_count,us_count,eu_count,other_count)]

year_list_citations = old_year_list_citations

print()
print('number_of_docs_witout_aff_country',number_of_docs_witout_aff_country)
print()
print("Average China citations: ",round(china_accumulated/china_count,2),' ',round(china_count),' ',round(china_total_citations))
for key_word in top_list:
    print(key_word, end = ' ')
    print(round(china_kw_citation_list[top_list.index(key_word)][1]/china_kw_citation_list[top_list.index(key_word)][2],2),' ',round(china_kw_citation_list[top_list.index(key_word)][2]),' ',round(china_kw_citation_list[top_list.index(key_word)][0]) )
print("Average US citations: ",round(us_accumulated/us_count,2),' ',round(us_count),' ',round(us_total_citations))
for key_word in top_list:
    print(key_word, end = ' ')
    print(round(us_kw_citation_list[top_list.index(key_word)][1]/us_kw_citation_list[top_list.index(key_word)][2],2),' ',round(us_kw_citation_list[top_list.index(key_word)][2]),' ',round(us_kw_citation_list[top_list.index(key_word)][0]) )
print("Average EU citations: ",round(eu_accumulated/eu_count,2),' ',round(eu_count),' ', round(eu_total_citations))
for key_word in top_list:
    print(key_word, end = ' ')
    print(round(eu_kw_citation_list[top_list.index(key_word)][1]/eu_kw_citation_list[top_list.index(key_word)][2],2),' ',round(eu_kw_citation_list[top_list.index(key_word)][2]),' ',round(eu_kw_citation_list[top_list.index(key_word)][0]) )
print("Average Other citations: ",round(other_accumulated/other_count,2),' ',round(other_count),' ', round(other_total_citations))
for key_word in top_list:
    print(key_word, end = ' ')
    print(round(other_kw_citation_list[top_list.index(key_word)][1]/other_kw_citation_list[top_list.index(key_word)][2],2),' ',round(other_kw_citation_list[top_list.index(key_word)][2]),' ',round(other_kw_citation_list[top_list.index(key_word)][0]) )
print("Average for top keywords")
for key_word in top_list:
    print(key_word, end = ' ')
    print(round((china_kw_citation_list[top_list.index(key_word)][1] + us_kw_citation_list[top_list.index(key_word)][1] + eu_kw_citation_list[top_list.index(key_word)][1] + other_kw_citation_list[top_list.index(key_word)][1])/(china_kw_citation_list[top_list.index(key_word)][2] + us_kw_citation_list[top_list.index(key_word)][2] + eu_kw_citation_list[top_list.index(key_word)][2] + other_kw_citation_list[top_list.index(key_word)][2]),2))
print("Average All citations: ",all_accumulated/all_count,' ',all_count)
print()

# accumulate results

current_age_factor = 1
total_dictionary = copy.deepcopy(my_dictionary)
age_dictionary = copy.deepcopy(my_dictionary)
citation_dictionary = copy.deepcopy(my_dictionary)
total_citation_dictionary = copy.deepcopy(my_dictionary)

ranking_dictionary = copy.deepcopy(my_dictionary)

for year_tup in year_list:
    for dict_key in (year_tup[1]):
        total_dictionary[dict_key].add_age_counter(year_tup[1][dict_key].get_count())
        total_dictionary[dict_key].make_union(year_tup[1][dict_key].get_set())
        
previous_year_tup = ()
for year_tup in year_list:
    if not year_tup[0] == years[0]: # not first year
        for dict_key in (year_tup[1]):
            age_dictionary[dict_key].add_age_counter((year_tup[1][dict_key].get_count() - previous_year_tup[1][dict_key].get_count())*current_age_factor)
    previous_year_tup = year_tup
    current_age_factor *= age_factor

for year_tup in year_list:
    for dict_key in (year_tup[1]):
        total_citation_dictionary[dict_key].add_age_counter(year_tup[1][dict_key].get_citations())

old_year_list_citations = copy.deepcopy(year_list_citations)

for year_tup in year_list:
    citation_tup = year_list_citations.pop(0)
    print("Citation tuple: ", citation_tup)
    for dict_key in (year_tup[1]):
        if total_dictionary[dict_key].get_age_counter() == 0:
            print("Error: ", dict_key)
        if year_tup[1][dict_key].get_count() > 0:
            citation_dictionary[dict_key].add_age_counter(((year_tup[1][dict_key].get_citations()/year_tup[1][dict_key].get_count())/citation_tup[3])*(year_tup[1][dict_key].get_count()/total_dictionary[dict_key].get_age_counter()))

year_list_citations = old_year_list_citations

# Print results

print()
print("Top 25 total")
sort_list = list(total_dictionary.items())
sort_list.sort(key = lambda x: x[1].get_age_counter(), reverse = True)
for dict_key in sort_list:
    ranking_dictionary[dict_key[0]].add_age_counter(sort_list.index(dict_key))
for i in range(25):
    for string in sort_list[i][0]:
        print(string, end = ' ')
    print(sort_list[i][1].get_age_counter(),' ',citation_dictionary[sort_list[i][0]].get_age_counter()) 

print()
print("Top 40 with age factor")
sort_list = list(age_dictionary.items())
sort_list.sort(key = lambda x: x[1].get_age_counter(), reverse = True)
for dict_key in sort_list:
    ranking_dictionary[dict_key[0]].add_age_counter(sort_list.index(dict_key))
for i in range(40):
    for string in sort_list[i][0]:
        print(string, end = ' ')
    print(sort_list[i][1].get_age_counter())

print()
print("Top 25 based on citations")
sort_list = list(total_citation_dictionary.items())
sort_list.sort(key = lambda x: x[1].get_age_counter(), reverse = True)
for dict_key in sort_list:
    ranking_dictionary[dict_key[0]].add_age_counter(sort_list.index(dict_key))
for i in range(25):
    for string in sort_list[i][0]:
        print(string, end = ' ')
    print(sort_list[i][1].get_age_counter()) 

'''    
  
top_list= []

print()
print("Top 20 based on combined ranking")
sort_list = list(ranking_dictionary.items())
sort_list.sort(key = lambda x: x[1].get_age_counter(), reverse = False)
for i in range(20):
    for string in sort_list[i][0]:
        print(string, end = ' ')
    print(sort_list[i][1].get_age_counter(),' ',citation_dictionary[sort_list[i][0]].get_age_counter())
    if i < 11:
        top_list += [sort_list[i][0]]
'''


print()
print("Top list")
print(top_list)

subset_of_list = []

print()
print("Subset of matrix")
for first_kw in top_list:
    for second_kw in top_list:
        format_float = "{:.2f}".format(len(total_dictionary[first_kw].get_set().intersection(total_dictionary[second_kw].get_set()))/len(total_dictionary[first_kw].get_set()))
        subset_of_list += [format_float]
        print(format_float, end = ' ')
    print()
    
print("Number of articles per keyword and year")

for first_kw in top_list:
    print(first_kw)
    for year_tup in year_list:
        print(year_tup[1][first_kw].get_count(), end =', ')
    print()

print()
old_china_count = 0
old_us_count = 0
old_eu_count = 0
old_other_count = 0

for country_tuple in year_list_countries:
    print('Year: ',country_tuple[0],' China ', country_tuple[1]-old_china_count,' North America ', country_tuple[2]-old_us_count,' EU ', country_tuple[3]-old_eu_count,' Other ', country_tuple[4]-old_other_count, )
    old_china_count = country_tuple[1]
    old_us_count = country_tuple[2]
    old_eu_count = country_tuple[3]
    old_other_count = country_tuple[4]