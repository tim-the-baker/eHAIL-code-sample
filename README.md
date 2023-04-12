# eHAIL-code-sample

I introduce a few abbreviations in rapid succession, so the following cheat sheet may be helpful. These abbreviations are also used in the code.
### Abbreviation cheat sheet:
RNS – random number source
PCC – probability conversion circuit
SNG – stochastic number generator (i.e., random bitstream generator)

**Context**: For my Ph.D., I studied circuits with random bitstreams as input. Those random bitstreams are generated by stochastic number generators (SNGs). An SNG consists of a random number source (RNS) and probability conversion circuit (PCC) that are connected (the RNS feeds into the PCC). There are a variety of RNS and PCC types and they can be combined in any manner. This organization lends itself well to object-oriented design.

**Code details**: The code contains three files, each containing a class that corresponding to these circuit modules. RNS is an abstract class that serves as a template for any type of random number source that we want to implement, e.g., LFSR_RNS. Likewise, PCC is an abstract class that serves as a template for probability conversion circuits. Finally, the SNG class seamlessly combines an RNS and PCC object and defines how they work together to generate random bitstreams (with the gen_SN method). Besides the abstract class, the code contains classes that implement a variety of specific RNS and PCC types that we study. 

The details of each class are not too important, but it is helpful to mention that we represent random bitstreams as PyTorch tensors where each entry corresponds to one bit. Once bits are generated by an SNG object, they are processed by other parts of the codebase that are not shown here. For example, we simulate neural networks that have random bitstreams as input. I wrote some skeleton code in the SNG file’s main function to show how the code is typically used.

**Why this code**: I chose to share this code because it is an exciting example of object-oriented design, an important skill for many software developers. The code also represents an example of progress. When I first started simulating circuits, the code was a bit of a mess. Over time patterns emerged and this object-oriented approach came to be. It has been instrumental in my Ph.D. research – we are able to study many variations of circuit designs in a streamlined manner. These three files are a small part of my simulation framework. Another part of the codebase relevant to e-HAIL is the dataflow management part which keeps track of each simulation’s parameters and its results.

Most relevant to e-HAIL, the code demonstrates that I am comfortable with the PyTorch Python library which is a popular deep learning framework. This code was originally written in NumPy (another Python library), but we switched when we started studying how these random bitstream circuits can be used to implement neural networks. The code also gives some examples of documentation which was written for my personal use (no one else has used this code).

