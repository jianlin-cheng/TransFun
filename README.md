# TransFun
Transformer for protein function prediction

# Uniprot
db|UniqueIdentifier|EntryName ProteinName OS=OrganismName OX=OrganismIdentifier [GN=GeneName ]PE=ProteinExistence SV=SequenceVersion
db is 'sp' for UniProtKB/Swiss-Prot and 'tr' for UniProtKB/TrEMBL.
UniqueIdentifier is the primary accession number of the UniProtKB entry.
EntryName is the entry name of the UniProtKB entry.
ProteinName is the recommended name of the UniProtKB entry as annotated in the RecName field. For UniProtKB/TrEMBL entries without a RecName field, the SubName field is used. In case of multiple SubNames, the first one is used. The 'precursor' attribute is excluded, 'Fragment' is included with the name if applicable.
OrganismName is the scientific name of the organism of the UniProtKB entry.
OrganismIdentifier is the unique identifier of the source organism, assigned by the NCBI.
GeneName is the first gene name of the UniProtKB entry. If there is no gene name, OrderedLocusName or ORFname, the GN field is not listed.
ProteinExistence is the numerical value describing the evidence for the existence of the protein.
SequenceVersion is the version number of the sequence.


