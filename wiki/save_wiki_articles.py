import wikipediaapi
from tqdm.auto import tqdm
from collections import Counter
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import pandas as pd
#
#wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 
#	                               'en', 
#									timeout=1000)
#
## extract pages
#train_pages = [
#	'Supersymmetric quantum mechanics',
#	'Relative density',
#	'Memristor',
#	'Quantization (physics)',
#	'Symmetry in biology',
#	'Mass versus weight',
#	'Navier–Stokes equations',
#	'Thermal equilibrium',
#	'Electrical resistivity and conductivity',
#	'Superconductivity',
#	'Black hole',
#	'Born reciprocity',
#	"Commentary on Anatomy in Avicenna's Canon",
#	'Supernova',
#	'Angular momentum',
#	'Condensation cloud',
#	'Minkowski space',
#	'Vacuum',
#	'Standard Model',
#	'Nebula',
#	'Antiferromagnetism',
#	'Light-year',
#	'Propagation constant',
#	'Phase transition',
#	'Redshift',
#	'The Ambidextrous Universe',
#	'Interstellar medium',
#	'Improper rotation',
#	'Plant',
#	'Clockwise',
#	'Morphology (biology)',
#	'Magnetic susceptibility',
#	'Nuclear fusion',
#	'Theorem of three moments',
#	'Lorentz covariance',
#	'Causality (physics)',
#	'Total internal reflection',
#	'Surgical pathology',
#	'Environmental Science Center',
#	'Electrochemical gradient',
#	'Planetary system',
#	'Cavitation',
#	'Parity (physics)',
#	'Dimension',
#	'Heat treating',
#	'Speed of light',
#	'Mass-to-charge ratio',
#	'Landau–Lifshitz–Gilbert equation',
#	'Point groups in three dimensions',
#	'Mammary gland',
#	'Convection (heat transfer)',
#	'Modified Newtonian dynamics',
#	"Earnshaw's theorem",
#	'Coherent turbulent structure',
#	'Phageome',
#	'Infectious tolerance',
#	'Ferromagnetism',
#	'Coffee ring effect',
#	'Magnetic resonance imaging',
#	'Ring-imaging Cherenkov detector',
#	'Tidal force',
#	'Kutta-Joukowski theorem',
#	'Radiosity (radiometry)',
#	'Quartz crystal microbalance',
#	'Crystallinity',
#	'Magnitude (astronomy)',
#	"Newton's law of universal gravitation",
#	'Uniform tilings in hyperbolic plane',
#	'Refractive index',
#	'Theorem',
#	'Leidenfrost effect',
#	'API gravity',
#	'Supersymmetry',
#	'Dark Matter',
#	'Molecular symmetry',
#	'Spin (physics)',
#	'Astrochemistry',
#	'List of equations in classical mechanics',
#	'Diffraction',
#	'C1 chemistry',
#	'Reciprocal length',
#	'Amplitude',
#	'Work function',
#	'Coherence (physics)',
#	'Ultraviolet catastrophe',
#	'Symmetry of diatomic molecules',
#	'Bollard pull',
#	'Linear time-invariant system',
#	'Triskelion',
#	'Cold dark matter',
#	'Frame-dragging',
#	"Fermat's principle",
#	'Enthalpy',
#	'Main sequence',
#	'QCD matter',
#	'Molecular cloud',
#	'Free neutron decay',
#	'Second law of thermodynamics',
#	'Droste effect',
#	'History of geology',
#	'Gravitational wave',
#	'Regular polytope',
#	'Spatial dispersion',
#	'Probability amplitude',
#	'Stochastic differential equation',
#	'Gravity Probe B',
#	'Electronic entropy',
#	'Renormalization',
#	'Unified field theory',
#	"Elitzur's theorem",
#	"Hesse's principle of transfer",
#	'Ecological pyramid',
#	'Virtual particle',
#	'Ramsauer–Townsend effect',
#	'Butterfly effect',
#	'Zero-point energy',
#	'Baryogenesis',
#	'Pulsar',
#	'Decay technique',
#	'Electric flux',
#	'Water hammer',
#	'Dynamic scaling',
#	'Luminance',
#	'Crossover experiment (chemistry)',
#	'Spontaneous symmetry breaking',
#	'Self-organization in cybernetics',
#	'Stellar classification',
#	'Probability density function',
#	'Pulsar-based navigation',
#	'Supermassive black hole',
#	'Explicit symmetry breaking',
#	'Surface power density',
#	'Organography',
#	'Copernican principle',
#	'Geometric quantization',
#	'Erlangen program',
#	'Magnetic monopole',
#	'Inflation (cosmology)',
#	'Heart',
#	'Observable universe',
#	'Wigner quasiprobability distribution',
#	'Shower-curtain effect',
#	'Scale (ratio)',
#	'Hydrodynamic stability',
#	'Paramagnetism',
#	'Emissivity',
#	'Critical Raw Materials Act',
#	'James Webb Space Telescope',
#	'Signal-to-noise ratio',
#	'Photophoresis',
#	'Time standard',
#	'Time',
#	'Galaxy',
#	'Rayleigh scattering'
#]
#
#print(f"{len(train_pages)} unique pages from train dataset")
#
#def get_wiki_sections_text(page):
#	ignore_sections = ["References", "See also", "External links", "Further reading", "Sources"]
#	wiki_page = wiki_wiki.page(page)
#	
#	# Get all the sections text
#	page_sections = [x.text for x in wiki_page.sections if x.title not in ignore_sections and x.text != ""]
#	section_titles = [x.title for x in wiki_page.sections if x.title not in ignore_sections and x.text != ""]
#	
#	# Add the summary page
#	page_sections.append(wiki_page.summary)
#	section_titles.append("Summary")
#	
#	return page_sections, section_titles
#
#
#def get_pages_df(pages):
#	page_section_texts = []
#	for page in tqdm(pages):
#		sections, titles = get_wiki_sections_text(page)
#		for section, title in zip(sections, titles):
#			page_section_texts.append({
#				'page': page,
#				'section_title': title,
#				'text': section
#			})
#	print(len(page_section_texts))
#	return pd.DataFrame(page_section_texts)
#
#train_pages_df = get_pages_df(train_pages)
#train_pages_df.to_csv("train_pages.csv", index=False)
#print(train_pages_df.shape)

train_pages_df = pd.read_csv('train_pages.csv')
train_pages_df.to_html('temp.html')
#print(train_pages_df.loc[8,'text'])
