# %%
from bs4 import BeautifulSoup
import urllib
import requests
from pyvis.network import Network
import networkx as nx
import argparse
import pickle
import scipy
import numpy as np
import json
import pandas as pd
import csv
import matplotlib.pyplot as plt

from collections import deque

INTERNAL_COLOR = '#0072BB'
EXTERNAL_COLOR = '#FF9F40'
ERROR_COLOR = '#FF0800'
RESOURCE_COLOR = '#2ECC71'

# %%
def handle_error(error, error_obj, r, url, visited, error_codes):
    error = str(error_obj) if error else r.status_code
    visited.add(url)
    error_codes[url] = error
    print(f'{error} ERROR while visiting {url}')

# %%
def crawl(url, visit_external):
    visited = set()
    edges = set()
    resource_pages = set()
    error_codes = dict()
    redirect_target_url = dict()

    head = requests.head(url, timeout=2)
    site_url = head.url
    redirect_target_url[url] = site_url

    to_visit = deque()
    to_visit.append((site_url, None))

    while to_visit:
        url, from_url = to_visit.pop()

        print('Visiting', url, 'from', from_url)

        error = False
        error_obj = None
        try:
            page = requests.get(url, timeout=2)
        except requests.exceptions.RequestException as e:
            error = True
            error_obj = e

        if error or not page:
            handle_error(error, error_obj, page, url, visited, error_codes)
            continue
        
        soup = BeautifulSoup(page.text, 'html.parser')
        internal_links = set()
        external_links = set()

        # Handle <base> tags
        base_url = soup.find('base')
        base_url = '' if base_url is None else base_url.get('href', '')

        for link in soup.find_all('a', href=True):
            link_url = link['href']
        
            if link_url.startswith('mailto:'):
                continue
            
            # Resolve relative paths
            if not link_url.startswith('http'):
                link_url = urllib.parse.urljoin(url, urllib.parse.urljoin(base_url, link_url))

            # Remove queries/fragments from internal links
            if link_url.startswith(site_url):
                link_url = urllib.parse.urljoin(link_url, urllib.parse.urlparse(link_url).path)

            # Load where we know that link_url will be redirected
            if link_url in redirect_target_url:
                link_url = redirect_target_url[link_url]

            if link_url not in visited and (visit_external or link_url.startswith(site_url)):
                is_html = False
                error = False
                error_obj = None

                try:
                    head = requests.head(link_url, timeout=2)
                    if head and 'html' in head.headers.get('content-type', ''):
                        is_html = True
                except requests.exceptions.RequestException as e:
                    error = True
                    error_obj = e

                if error or not head:
                    handle_error(error, error_obj, head, link_url, visited, error_codes)
                    edges.add((url, link_url))
                    continue

                redirect_target_url[link_url] = head.url
                link_url = head.url
                visited.add(link_url)

                if link_url.startswith(site_url):
                    if is_html:
                        to_visit.append((head.url, url))
                    else:
                        resource_pages.add(link_url)
            
            edges.add((url, link_url))
    
    return edges, error_codes, resource_pages

# %%
def get_node_info(nodes, error_codes, resource_pages, args):
    node_info = []
    for node in nodes:
        if node in error_codes:
            node_info.append(f'Error: {error_codes[node]}')
        elif node in resource_pages:
            node_info.append('resource')
        elif node.startswith(args.site_url):
            node_info.append('internal')
        else:
            node_info.append('external')
    return node_info

# %%
def visualize(edges, error_codes, resource_pages, args):
    G = nx.DiGraph()
    
    G.add_edges_from(edges) 
    pr1 = nx.pagerank(G)#, alpha=1, max_iter=500, tol=1e-15)
    
    F = nx.DiGraph()
    F.add_edges_from(edges)
    S = nx.convert_node_labels_to_integers(F)
    pr2 = nx.pagerank(S)#, alpha=1, max_iter=500, tol=1e-15)
    with open("pagerank_2.txt", 'w') as f: 
        for key, value in pr2.items(): 
            f.write('%s:%s\n' % (key,value))
    
    with open("pagerank.txt", 'w') as f: 
        for key, value in pr1.items(): 
            f.write('%s:%s\n' % (key,value))
    
    with open("pagerank_1.txt", 'w') as f: 
        for key, value in pr1.items(): 
            f.write('%s\n' % (value))
            
    if args.save_txt is not None or args.save_npz is not None:
        nodes = list(G.nodes())
        adj_matrix = nx.to_numpy_matrix(G, nodelist=nodes, dtype=int)

        if args.save_npz is not None:
            base_fname = args.save_npz.replace('.npz', '')
            scipy.sparse.save_npz(args.save_npz, scipy.sparse.coo_matrix(adj_matrix))
        else:
            base_fname = args.save_txt.replace('.txt', '')
            np.savetxt(args.save_txt, adj_matrix, fmt='%d')

        node_info = get_node_info(nodes, error_codes, resource_pages, args)
        with open(base_fname + '_nodes.txt', 'w') as f:
            f.write('\n'.join([nodes[i] + '\t' + node_info[i] for i in range(len(nodes))]))

    net = Network(width=args.width, height=args.height) #directed = True - arrows
    net.from_nx(G)

    
    net.show_buttons()
    if args.options is not None:
        try:
            with open(args.options, 'r') as f:
                net.set_options(f.read())
        except FileNotFoundError as e:
            print('Error: options file', args.options, 'not found.')
        except Exception as e:
            print('Error applying options:', e)
    
    for node in net.nodes:
        node['size'] = 15
        node['label'] = ''
        #node['physics'] = "False"
        
        if node['id'].startswith(args.site_url):
            node['color'] = INTERNAL_COLOR
            if node['id'] in resource_pages:
                node['color'] = RESOURCE_COLOR
        else:
            node['color'] = EXTERNAL_COLOR
        
        if node['id'] in error_codes:
            node['title'] = f'{error_codes[node["id"]]} Error: <a href="{node["id"]}">{node["id"]}</a>'
            node['color'] = ERROR_COLOR
        else:
            node['title'] = f'<a href="{node["id"]}">{node["id"]}</a>'

    #net.toggle_physics(False)
    net.save_graph(args.vis_file)
    #net.show(args.vis_file, local=True)

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the link graph of a website.')
    parser.add_argument('site_url', type=str, help='the base URL of the website', nargs='?', default='')
    default = 'site.html'
    parser.add_argument('--vis-file', type=str, help='filename in which to save HTML graph visualization (default: ' + default + ')', default=default)
    default = 'crawl.pickle'
    parser.add_argument('--data-file', type=str, help='filename in which to save crawled graph data (default: ' + default + ')', default=default)
    default = 1500
    parser.add_argument('--width', type=int, help='width of graph visualization in pixels (default: ' + str(default) + ')', default=default)
    default = 1200
    parser.add_argument('--height', type=int, help='height of graph visualization in pixels (default: ' + str(default) + ')', default=default)
    parser.add_argument('--visit-external', action='store_true', help='detect broken external links (slower)')
    parser.add_argument('--show-buttons', action='store_true', help='show visualization settings UI')
    parser.add_argument('--options', type=str, help='file with drawing options (use --show-buttons to configure, then generate options)')
    parser.add_argument('--from-data-file', type=str, help='create visualization from given data file', default=None)
    parser.add_argument('--force', action='store_true', help='override warnings about base URL')
    parser.add_argument('--save-txt', type=str, nargs='?', help='filename in which to save adjacency matrix (if no argument, uses adj_matrix.txt). Also saves node labels to [filename]_nodes.txt', const='adj_matrix.txt', default=None)
    parser.add_argument('--save-npz', type=str, nargs='?', help='filename in which to save sparse adjacency matrix (if no argument, uses adj_matrix.npz). Also saves node labels to [filename]_nodes.txt',  const='adj_matrix.npz', default=None)

    args = parser.parse_args(args=[])
    #args = vars(parser.parse_args(args=[]))
    args.site_url = #put here any site you want to scrap
    if args.from_data_file is None:
        if not args.site_url.endswith('/'):
            if not args.force:
                print('Warning: no trailing slash on site_url (may get duplicate homepage node). If you really don\'t want the trailing slash, run with --force')
                exit(1)

        if not args.site_url.startswith('https'):
            if not args.force:
                print('Warning: not using https. If you really want to use http, run with --force')
                exit(1)
        
        edges, error_codes, resource_pages = crawl(args.site_url, args.visit_external)
        print('Crawl complete.')

        with open(args.data_file, 'wb') as f:
            pickle.dump((edges, error_codes, resource_pages, args.site_url), f)
            print(f'Saved crawl data to {args.data_file}')
    else:
        with open(args.from_data_file, 'rb') as f:
            edges, error_codes, resource_pages, site_url = pickle.load(f)
            args.site_url = site_url

    cw = csv.writer(open("hello.csv",'w'))
    
    
    df = pd.DataFrame(list(edges), columns=["colummn1","colummn2"])
    df.to_csv('list.csv', index=False)

    #print(list(edges))
    #cw.writerow(list(edges))
   
    #print(edges)
    visualize(edges, error_codes, resource_pages, args)
    print('Saved graph to', args.vis_file)

# %%
df = pd.DataFrame(list(edges), columns=["colummn1","colummn2"])
print(df)

# %%
F = nx.DiGraph()
F.add_edges_from(edges)

S = nx.convert_node_labels_to_integers(F)
nodes_F = list(S.nodes())
edges_F = list(S.edges())

#print(edges)
df1 = pd.DataFrame(list(S.edges()), columns=["colummn1","colummn2"])
df1.to_csv('list_integer.csv', index=False)
#print(df1)

pr1 = nx.pagerank(S)#, alpha=1, max_iter=500, tol=1e-15)
for key, value in pr1.items(): 
    print('%s:%s\n' % (key,value))

# %%
pr1_sorted = sorted(pr1.items(), key=lambda x:x[1], reverse = True)

# %%
pr1_sorted

# %%
G = nx.DiGraph()
G.add_edges_from(data)
G = nx.convert_node_labels_to_integers(G)
print(len(G.edges()))
simple_pagerank = nx.pagerank(G, alpha=1, max_iter=500, tol=1e-15)
pr2 = nx.pagerank(G)#, alpha=1, max_iter=500, tol=1e-15)
for key, value in pr2.items(): 
    print('%s:%s\n' % (key,value))
pr2_sorted = sorted(pr2.items(), key=lambda x:x[1], reverse = True)
print(pr2_sorted)
hubs, authorities = nx.hits(G)
for key, value in authorities.items():
    print('%s:%s' % (key,value))
hubs_sorted = sorted(hubs.items(), key=lambda x:x[1], reverse = True)
print(hubs_sorted)
authorities_sorted = sorted(authorities.items(), key=lambda x:x[1], reverse = True)
print(authorities_sorted)

# %% [markdown]
# # Sample

# %%
g = net.Network(notebook = True, width = 1000, height = 800, directed = True)

G = nx.DiGraph()

[G.add_node(k) for k in ["1", "2", "3", "4", "5", "6"]]
G.add_edges_from([('1','2'),('1','3'), ('1','6'),
                  ('2','4'),
                  ('3','1'),('3','2'),('3','4'),
                  ('4','5'),('4','6'),
                  ('5','4'),('5','6'),
                  ('6','3'),('6','5')
                 ])

simple_pagerank = nx.pagerank(G, alpha=1, max_iter=500, tol=1e-15)

g.from_nx(G)
g.show("example.html")

simple_pagerank = nx.pagerank(G)

print(simple_pagerank)

h, a = nx.hits(G)
print(h)
print(a)


