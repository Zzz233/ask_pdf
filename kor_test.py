from langchain.chat_models import ChatOpenAI
from kor import create_extraction_chain, Object, Text, Number
from langchain.callbacks import get_openai_callback

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key="",
    temperature=0,
    max_tokens=2000,
    frequency_penalty=0,
    presence_penalty=0,
    top_p=1.0,
)


parts = Object(
    id="relationship",
    description="Relationship between two parts",
    attributes=[
        Text(id="part", description="The name of the part")
    ],
    examples=[
        (
            "the jeep has wheels and windows",
            [
                {"part": "wheel"},
                {"part": "window"}
            ],
        )
    ]
)


schema = Object(
    id="sentence",
    description=(
        "Extract the relationship among gene, protein, mRNA, lncRNA, miRNA, or circRNA in the given sentence."
    ),
    attributes=[
        Text(
            id="gene",
            description="Gene's common name",
        ),
        Text(
            id="protein",
            description="Protein's common name",
        ),
        Text(
            id="mrna",
            description="mRNA's common name",
        ),
        Text(
            id="incrna",
            description="IncRNA's common name",
        ),
        Text(
            id="mirna",
            description="miRNA's common name",
        ),
        Text(
            id="circrna",
            description="circRNA's common name",
        ),
    ],
    examples=[
        (
            "CDR1as depletion results from epigenetic silencing of LINC00632, its originating long non-coding RNA (lncRNA) and promotes invasion in vitro and metastasis in vivo through a miR-7-independent, IGF2BP3-mediated mechanism, Additionally, we discovered a regulatory role of the circ 0006528-miR-7-5p-Raf1 axis in ADMresistant breast cancer, both of these domains have striking sequence homology with human SIM mRNA and Drosophila SIM protein.",
            [
                {"gene": ["CDR1as", "IGF2BP3", "Raf1", "SIM"]},
                {"protein": ["CDR1as", "IGF2BP3", "Raf1", "SIM"]},
                {"mran": ["CDR1as", "IGF2BP3", "Raf1", "SIM"]},
                {"incrna": ["LINC00632"]},
                {"mirna": ["miR-7", "miR-7-5p"]},
                {"circrna": ["circ 0006528"]},
            ]
        )
    ]
)

chain = create_extraction_chain(llm, schema, encoder_or_encoder_class='json')

prompt = chain.prompt.format_prompt(text="We showed earlier that VHL downregulates vascular endothelial growth factor transcription by directly binding and inhibiting the transcriptional activator Sp1.").to_string()
print(prompt)
# with get_openai_callback() as cb:
#     resp = chain.predict_and_parse(
#         text="We showed earlier that VHL downregulates vascular endothelial growth factor transcription by directly binding and inhibiting the transcriptional activator Sp1.")['data']
#     print(resp)
#     print(f"Total Tokens: {cb.total_tokens}")
#     print(f"Prompt Tokens: {cb.prompt_tokens}")
#     print(f"Completion Tokens: {cb.completion_tokens}")
#     print(f"Total Cost (USD): ${cb.total_cost}")


"""
# id, effect_Regulation_Pattern_rectify, sentence

'regulation', 'The transcription factors Sp1, [E1]Sp3[/E1], and AP-2 are required for constitutive [E2]matrix metalloproteinase-2[/E2] gene expression in astroglioma cells.'

'no-related', 'BACKGROUND & AIMS: The intestine-specific caudal-related [E2]homeobox[/E2] transcription factor [E1]CDX2[/E1] seems to play a key role in intestinal
               development and differentiation.'

'activation', 'Notably, we showed that [E1]YY1[/E1] enhances AP-2alpha transcriptional activation of the [E2]ERBB2[/E2] promoter through an AP-2 site both in HepG2 and
               in HCT116 cells, whereas a carboxyl-terminal-truncated form of YY1 cannot.'

'repression', 'Analysis of genes that could be potential targets of KLF2 revealed that [E1]KLF2[/E1] negatively regulated [E2]WEE1[/E2] expression.'

'ppiregulation', 'Deletion of the 96-122 domain prevents [E1]VHL[/E1] effects on [E2]Sp1[/E2] DNA binding and on VHL target gene expression,
                  indicating the domain contributes importantly to VHL tumor suppressor activity.'

'ppiactivation', 'Notably, we showed that [E1]YY1[/E1] enhances [E2]AP-2alpha[/E2] transcriptional activation of the ERBB2 promoter through an AP-2 site both
                  in HepG2 and in HCT116 cells, whereas a carboxyl-terminal-truncated form of YY1 cannot.'

'ppirepression', 'We showed earlier that [E1]VHL[/E1] downregulates vascular endothelial growth factor transcription by directly binding and inhibiting the
                  transcriptional activator [E2]Sp1[/E2].'
"""