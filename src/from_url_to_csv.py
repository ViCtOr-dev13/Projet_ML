from urllib.parse import urlparse
import pandas as pd
import re
import os
from itertools import groupby
from collections import Counter
import math



def count_chars(s, chars):
    return sum(s.count(c) for c in chars)

def entropy(s):
    from collections import Counter
    import math
    p, lns = Counter(s), float(len(s))
    return -sum( count/lns * math.log(count/lns, 2) for count in p.values())

def parse_url(url):
    parsed = urlparse(url)

    querylength = len(parsed.query)

    domain = parsed.netloc
    domain_tokens = domain.split('.')
    domain_token_count = len(domain_tokens)

    path = parsed.path
    path_tokens = path.split('/')
    path_token_count = len(path_tokens) - 1

    avgdomaintokenlen = sum(len(token) for token in domain_tokens) / domain_token_count if domain_token_count else 0

    longdomaintokenlen = max(len(token) for token in domain_tokens) if domain_tokens else 0

    avgpathtokenlen = sum(len(token) for token in path_tokens) / path_token_count if path_token_count else 0

    tld = domain_tokens[-1] if domain_tokens else ''

    charcompvowels = count_chars(url, 'aeiou')
    charcompace = count_chars(url, ' ')

    url_len = len(url)
    domain_len = len(domain)
    path_len = len(path)

    is_port_80 = 1 if parsed.port == 80 else 0

    num_dots = url.count('.')

    entropy_url = entropy(url)

    filename = parsed.path.split('/')[-1] if parsed.path else ''
    filename_len = len(filename)

    ext_len = len(parsed.path.split('.')[-1]) if '.' in parsed.path else 0

    query_letter_count = sum(c.isalpha() for c in parsed.query)
    query_digit_count = sum(c.isdigit() for c in parsed.query)

    spcharUrl = len(re.findall(r'\W', url))

    delimeter_Domain = domain.count('-')
    delimeter_path = path.count('-')
    delimeter_Count = url.count('-')

    NumberRate_URL = sum(c.isdigit() for c in url) / len(url)
    NumberRate_Domain = sum(c.isdigit() for c in domain) / len(domain) if domain else 0

    directory = path.split('/')[1] if '/' in path else ''
    NumberRate_DirectoryName = sum(c.isdigit() for c in directory) / len(directory) if directory else 0

    filename = path.split('/')[-1] if '/' in path else ''
    File_name_DigitCount = sum(c.isdigit() for c in filename)

    ext = os.path.splitext(path)[1]
    Extension_DigitCount = sum(c.isdigit() for c in ext)

    directory = path.split('/')[1] if '/' in path else ''
    Directory_DigitCount = sum(c.isdigit() for c in directory)

    URL_DigitCount = sum(c.isdigit() for c in url)

    host = parsed.netloc
    host_DigitCount = sum(c.isdigit() for c in host)
    URL_Letter_Count = sum(c.isalpha() for c in url)

    host = parsed.netloc
    host_letter_count = sum(c.isalpha() for c in host)

    directory = path.split('/')[1] if '/' in path else ''
    Directory_LetterCount = sum(c.isalpha() for c in directory)

    filename = path.split('/')[-1] if '/' in path else ''
    Filename_LetterCount = sum(c.isalpha() for c in filename)

    ext = os.path.splitext(path)[1]
    Extension_LetterCount = sum(c.isalpha() for c in ext)

    ISIpAddressInDomainName = 1 if host.replace('.', '').isdigit() else 0

    # Characteres specials URL
    special_chars = "!@#$%^&*()_+-={}[]|:;'<>?,./\""
    spcharUrl = sum(c in special_chars for c in url)

    delimeter_Domain = sum(c in special_chars for c in parsed.netloc)
    delimeter_path = sum(c in special_chars for c in parsed.path)

    NumberRate_URL = sum(c.isdigit() for c in url) / len(url)
    NumberRate_Domain = sum(c.isdigit() for c in parsed.netloc) / len(parsed.netloc)

    directory_name = parsed.path.split('/')[1] if '/' in parsed.path else ''
    NumberRate_DirectoryName = sum(c.isdigit() for c in directory_name) / len(directory_name) if directory_name else 0

    LongestPathTokenLength = max(len(token) for token in path_tokens)

    NumberofDotsinURL = url.count('.')

    URL_sensitiveWord = 'sensitive' in url.lower()

    file_name = parsed.path.split('/')[-1]
    NumberRate_FileName = sum(c.isdigit() for c in file_name) / len(file_name) if file_name else 0

    extension = os.path.splitext(parsed.path)[1]
    #NumberRate_Extension = sum(c.isdigit() for c in extension) / len(extension) if extension else 0

    after_path = url.split(parsed.path, 1)[-1]
    NumberRate_AfterPath = sum(c.isdigit() for c in after_path) / len(after_path) if after_path else 0

    SymbolCount_URL = sum(not c.isalnum() for c in url)
    SymbolCount_Domain = sum(not c.isalnum() for c in parsed.netloc)

    entropy_url = -sum(p * math.log(p) / math.log(2.0) for p in Counter(url).values())
    entropy_domain = -sum(p * math.log(p) / math.log(2.0) for p in Counter(parsed.netloc).values())

    directory_name = parsed.path.split('/')[-1]
    entropy_directoryname = -sum(p * math.log(p) / math.log(2.0) for p in Counter(directory_name).values())

    filename = os.path.splitext(parsed.path.split('/')[-1])[0]
    entropy_filename = -sum(p * math.log(p) / math.log(2.0) for p in Counter(filename).values())

    extension = os.path.splitext(parsed.path)[1]
    entropy_extension = -sum(p * math.log(p) / math.log(2.0) for p in Counter(extension).values())


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


    ldl_url = len(parsed.netloc) + len(parsed.path) + len(parsed.query)
    ldl_domain = len(parsed.netloc)
    ldl_path = len(parsed.path)
    ldl_filename = len(parsed.path.split('/')[-1]) if '/' in parsed.path else 0
    ldl_getArg = len(parsed.query)

    dld_url = len(url) - len(parsed.netloc) - len(parsed.path) - len(parsed.query)
    dld_domain = len(parsed.netloc) - parsed.netloc.count('.')
    dld_path = len(parsed.path) - parsed.path.count('/')
    dld_filename = len(parsed.path.split('/')[-1]) if '/' in parsed.path else 0
    dld_getArg = len(parsed.query) - parsed.query.count('&')

    subDirLen = sum(len(token) for token in parsed.path.split('/')[1:-1])
    ArgLen = len(parsed.query)
    pathurlRatio = len(parsed.path) / len(url) if len(url) > 0 else 0
    ArgUrlRatio = len(parsed.query) / len(url) if len(url) > 0 else 0
    argDomanRatio = len(parsed.query) / len(parsed.netloc) if len(parsed.netloc) > 0 else 0
    domainUrlRatio = len(parsed.netloc) / len(url) if len(url) > 0 else 0
    pathDomainRatio = len(parsed.path) / len(parsed.netloc) if len(parsed.netloc) > 0 else 0
    argPathRatio = len(parsed.query) / len(parsed.path) if len(parsed.path) > 0 else 0

    executable = 1 if parsed.path.endswith(('.exe', '.dll', '.bat', '.scr', '.cmd')) else 0

    chars = parsed.netloc + parsed.path + parsed.query
    continuity_counts = [len(list(group)) for key, group in groupby(chars)]
    character_continuity_rate = max(continuity_counts) / len(chars) if len(chars) > 0 else 0

    variables = re.findall(r'\w+=\w+', url)
    longest_variable_value = max([len(var.split('=')[1]) for var in variables]) if variables else 0

    url_digit_count = sum(c.isdigit() for c in url)
    host_digit_count = sum(c.isdigit() for c in parsed.netloc)
    directory_digit_count = sum(c.isdigit() for c in parsed.path.split('/')[1]) if '/' in parsed.path else 0

    filename_digit_count = sum(c.isdigit() for c in parsed.path.split('/')[-1]) if '/' in parsed.path else 0
    extension_digit_count = sum(c.isdigit() for c in os.path.splitext(parsed.path)[1]) if '.' in parsed.path else 0

    domain_longest_word_length = max(len(word) for word in parsed.netloc.split('.')) if '.' in parsed.netloc else 0
    path_longest_word_length = max(len(word) for word in parsed.path.split('/')) if '/' in parsed.path else 0
    sub_directory_longest_word_length = max(len(word) for word in parsed.path.split('/')[1:-1]) if '/' in parsed.path and len(parsed.path.split('/')[1:-1]) > 0 else 0
    arguments_longest_word_length = max(len(word.split('=')[1]) for word in parsed.query.split('&')) if '&' in parsed.query else 0

    url_queries_variable = len(variables)
    delimeter_count = sum(c in ['-', '_', '~', '.'] for c in chars)
    number_rate_filename = sum(c.isdigit() for c in parsed.path.split('/')[-1]) / len(parsed.path.split('/')[-1]) if '/' in parsed.path else 0
    symbol_count_directoryname = sum(not c.isalnum() for c in parsed.path.split('/')[1]) if '/' in parsed.path else 0

    symbol_count_filename = sum(not c.isalnum() for c in parsed.path.split('/')[-1]) if '/' in parsed.path else 0
    symbol_count_extension = sum(not c.isalnum() for c in os.path.splitext(parsed.path)[1]) if '.' in parsed.path else 0
    symbol_count_afterpath = sum(not c.isalnum() for c in url.split(parsed.path, 1)[-1]) if parsed.path else 0

    # Entropy_Afterpath

    # URL_Type_obf_Type
    # Déterminer si une URL est de type "obf" (obfusquée) besoin d'une base de données de sites malveillants ou  méthode d'apprentissage automatique pour classer l'URL
    return [querylength, domain_token_count, path_token_count, avgdomaintokenlen, longdomaintokenlen, avgpathtokenlen, tld, charcompvowels, charcompace, url_len, domain_len, path_len, is_port_80, num_dots, entropy_url, filename_len, ext_len, query_letter_count, query_digit_count, spcharUrl, delimeter_Domain, delimeter_path, NumberRate_URL, NumberRate_Domain, NumberRate_DirectoryName, URL_Letter_Count, host_letter_count, Directory_LetterCount, Filename_LetterCount, Extension_LetterCount, ISIpAddressInDomainName, LongestPathTokenLength, NumberofDotsinURL, URL_sensitiveWord, NumberRate_FileName, '''NumberRate_Extension''', NumberRate_AfterPath, SymbolCount_URL, SymbolCount_Domain, entropy_url, entropy_domain, entropy_directoryname, entropy_filename, entropy_extension, ldl_url, ldl_domain, ldl_path, ldl_filename, ldl_getArg, dld_url, dld_domain, dld_path, dld_filename, dld_getArg, subDirLen, ArgLen, pathurlRatio, ArgUrlRatio, argDomanRatio, domainUrlRatio, pathDomainRatio, argPathRatio, executable, character_continuity_rate, longest_variable_value, url_digit_count, host_digit_count, directory_digit_count, filename_digit_count, extension_digit_count, domain_longest_word_length, path_longest_word_length, sub_directory_longest_word_length, arguments_longest_word_length, url_queries_variable, delimeter_count, number_rate_filename, symbol_count_directoryname, symbol_count_filename, symbol_count_extension, symbol_count_afterpath]

data = []
url = "http://www.sinduscongoias.com.br/index.html"
data.append(parse_url(url))

df = pd.DataFrame(data, columns=["Querylength", "domain_token_count", "path_token_count", "avgdomaintokenlen", "longdomaintokenlen", "avgpathtokenlen", "tld", "charcompvowels", "charcompace", "urlLen", "domainlength", "pathLength", "isPortEighty", "NumberofDotsinURL", "Entropy_URL", "fileNameLen", "this.fileExtLen", "Query_LetterCount", "Query_DigitCount", "spcharUrl", "delimeter_Domain", "delimeter_path", "NumberRate_URL", "NumberRate_Domain", "NumberRate_DirectoryName", "URL_Letter_Count", "host_letter_count", "Directory_LetterCount", "Filename_LetterCount", "Extension_LetterCount", "ISIpAddressInDomainName", "LongestPathTokenLength", "NumberofDotsinURL", "URL_sensitiveWord", "NumberRate_FileName", '''"NumberRate_Extension"''', "NumberRate_AfterPath", "SymbolCount_URL", "SymbolCount_Domain", "Entropy_URL", "Entropy_Domain", "Entropy_DirectoryName", "Entropy_Filename", "Entropy_Extension", "ldl_url", "ldl_domain", "ldl_path", "ldl_filename", "ldl_getArg", "dld_url", "dld_domain", "dld_path", "dld_filename", "dld_getArg", "subDirLen", "ArgLen", "pathurlRatio", "ArgUrlRatio", "argDomanRatio", "domainUrlRatio", "pathDomainRatio", "argPathRatio", "executable", "character_continuity_rate", "longest_variable_value", "url_digit_count", "host_digit_count", "directory_digit_count", "filename_digit_count", "extension_digit_count", "domain_longest_word_length", "path_longest_word_length", "sub_directory_longest_word_length", "arguments_longest_word_length", "url_queries_variable", "delimeter_count", "number_rate_filename", "symbol_count_directoryname", "symbol_count_filename", "symbol_count_extension", "symbol_count_afterpath"])
print(df)

df.to_csv('urls.csv', index=False)


