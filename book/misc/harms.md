(hazards)=
# List of Harms
There are many ways in which machine learning systems can be harmful. This page contains a (non-exhaustive) list of hazards and harms, organized by the moral value the harm threatens.

```{admonition} Attribution
This list was inspired by and adapted from Microsoft's [harms modeling framework](https://learn.microsoft.com/en-us/azure/architecture/guide/responsible-innovation/harms-modeling/type-of-harm).
```

## Safety
| Hazard | Harm | Description | Considerations | Example | Resources |
| - | - | - | - | - | - |
| Lacking accuracy | Injury | System errors can lead to physical, emotional, or psychological injury. | How do system errors impact people? | An ilness is misdiagnosed, leading to unnecessary treatment. | {footcite:t}`raji2022fallacy` |
| Inadequate testing | Injury | Real-world testing of system failure modes insufficiently considers a diverse set of users and scenarios. | Are all failure modes adequately tested? How are users impacted by a system failure? Are human interventions possible? | An autonomous vehicle that is tested in the public sphere for new data collection kills a pedestrian, because the system did not recognize the pedestrian crossing outside of a sidewalk. | {footcite:t}`raji2022fallacy` |

## Fairness
| Hazard | Harm | Description | Considerations | Example | Resources |
| - | - | - | - | - | - |
| Social bias | {term}`Allocation harm` | The system results in unfair allocation of opportunities, resources, or information and amplification of power inequality. | Are some groups allocated fewer opportunities than others? | A resume screening application consistently ranks male candidates higher than female candidates. | [Algorithmic Fairness](intro_fairness) |
| Social bias | {term}`Quality-of-service harm` | The system disproportionately fails for certain groups of people. |  Does the system work equally well for different demographics, particularly those defined by sensitive characterises? | A facial recognition system disproportionately misidentifies faces of black women compared to white men {footcite:p}`buolamwini2018` | [Algorithmic Fairness](intro_fairness) |
| Social bias | {term}`Stereotyping harm` | The system reinforces negative societal stereotypes. | Does the system return output that reinforces undesirable stereotypes? | A search engine returns only pictures of men when prompted with "CEO", reinforcing stereotypes. | [Algorithmic Fairness](intro_fairness) |
| Social bias | {term}`Denigration harm` | The system is actively derogatory or offensive. | In what ways could the system output be considered offensive for stakeholders? | An image tagging system tags a photo of black people as an animal. | [Algorithmic Fairness](intro_fairness) |
| Social bias | {term}`Representation harm` | The development or usage of the system over- or under-represents certain groups of people. | Are some groups overburdened compared to others, even if they do not benefit from the system? | Residents of an indigenous reservation have been subject to intense biomedical data collection, but this has not yielded any significant improvements in health outcomes amongst community members {footcite:p}`paullada2020data` | [Algorithmic Fairness](intro_fairness) | 
| Feature Selection and Opaque Decision-Making | {term}`Procedural harm` | The system uses features that are arbitrary or otherwise violate social norms. | Are the features relevant to the problem at hand? Are they used in fair ways? | An arbitrary feature is included in a resume selection application. | {footcite}`Rudin2018a` |

## Transparency
| Hazard | Harm | Description | Considerations | Example | Resources |
| - | - | - | - | - | - |
| Opaque Automated Decision-Making | Loss of Effective Remedy | An inability to explain the rationale or lack of opportunity a contest a decision. | How might people understand the reasoning for decisions made by this technology? How might an individual that relies on this technology explain the decisions it makes? How could people contest or question a decision this technology makes? | Automated prison sentence or pre-trial release decision is not explained to the accused. |  |
| Opaque Automated Decision-Making | Misguided Trust in Automation | Misguided beliefs can lead users to overtrust the reliability of a digital agent. | How could sole dependence on an artificial agent impact a person? | A chat bot could be relied upon for relationship advice or mental health counseling instead of a trained professional. |  |

## Accountability
| Hazard | Harm | Description | Considerations | Example | Resources |
| - | - | - | - | - | - |
| Opaque Automated Decision-Making | Lack of Accountability | Opaque decision-making related to the development and usage of a machine learning system hinders accountability. | Is the development process transparent? Can decisions be explained? Is it clear who is responsible for what aspects of the system? | A machine learning model is used to reject lean applications and responsibility is transferred to the incomprehensibility of the model. | |
| No provisions for auditing | Lack of accountability | Third parties are unable to review the behavior of an algorithm. | Are detailed documentation and technically suitable API's available and to whom? | A large social media platform does not provide documentation or suitable APIs to audit fairness of their algorithms. |  |
| No mechanisms for recourse | Lack of redress | Stakeholders are unable to ask for compensation or reparation for an undesirable decision or situation. | Who is responsible if users are harmed by this product? What will the reporting process and process for recourse be? | A user is banned from a social media platform and is unable to contact the platform to contest the decision. |  |

## Privacy
| Hazard | Harm | Description | Considerations | Example | Resources |
| - | - | - | - | - | - |
| Insecure and/or Redundant Data Collection, Storage, Aggregation, and Processing | Loss of Data Privacy | Reveal personal information a person has not consented to share. | How could this technology use information to infer portions of a private life? How could decisions based upon these inferences expose things that a person does not want made public?	 | A language model trained on large amounts of data reveals personally identifiable information when queried for specific examples {footcite:p}`carlini2020extracting` |  |
| Lack of a data retention policy | Never forgiven | Digital files or records are never deleted. | What and where is data being stored from this product, and who can access it? How long is user data stored after technology interaction? How is user data updated or deleted? | A teenager's social media history could remain searchable long after they have outgrown the platform. |  |
| Adversarial use | Identity Theft | Loss of control over personal credentials, reputation, and/or representation. | How might an individual be impersonated with this technology? How might this technology mistakenly recognize the wrong individual as an authentic user? | A synthetic voice could mimic the sound of a person's voice and be used to access a bank account. |  |
| Data leak | Public Shaming | Reveal private, sensitive, or socially inappropriate information.  |  How might movements or actions be revealed through data aggregation? | A fitness app could reveal a user's GPS location on social media. |  |
| Surveillance | Loss of Freedom of Movement with Desired Anonymity | An inability to navigate the physical or virtual world with desired anonymity.|  In what ways might this technology monitor people across physical and virtual spaces? | A facial recognition system is used to monitor civilians.|  |

## Liberty
| Hazard | Harm | Description | Considerations | Example | Resources |
| - | - | - | - | - | - |
| Forced Participation and Surveillance | Loss of Freedom of Movement | Requiring participation in the use of technology or surveillance to take part in society. | How might use of this technology be required for participation in society or organization membership? | Biometric enrollment in a company's meeting room transcription AI is a stipulated requirement in job offer letter. | |
| Forced Social Control | Inability to Fully Develop Personality | Reinforcing conformity and connotations towards particular personality traits. | What types of personal or behavioral data might feed this technology? How would it be obtained? What outputs would be derived from this data? Is this technology likely to be used to encourage or discourage certain behaviors? | Authoritarian government uses social media and e-commerce data to determine a "trustworthiness" score based on where people shop and who they spend time with. |  |
|     Limited Expression of Perspectives | Loss of Freedom of Expression | Amplification of majority opinions and an inability to express a unique perspective. | How might this technology amplify majority opinions or "group-think"? Conversely, how might unique forms of expression be suppressed. In what ways might the data gathered by this technology be used in feedback to people? | Limited options for gender in an automated loan application system inhibits self-expression of a person's diversity. | Resources |

## Sustainability
| Hazard | Harm | Description | Considerations | Example | Resources |
| - | - | - | - | - | - |
| Physical System Components | Electronic Waste | Reduced quality of collective well-being because of the inability to repair, recycle, or otherwise responsibly dispose of electronics.  | How might this technology reduce electronic waste by recycling materials or allowing users to self-repair? How might this technology contribute to electronic waste when new versions are released or when current/past versions stop working? | Toxic materials inside discarded electronic devices could leach into the water supply, making local populations ill. | |
| Physical System Components | Exploitation or Depletion of Resources | Obtaining the raw materials for a technology, including how it's powered, leads to negative consequences to the environment and its inhabitants. | What materials are needed to build or run this technology? What energy requirements are needed to build or run this technology?	 | Large scale data storage and computation cost invite climate abuse {footcite:p}`Raji2020a`. |  |

## Autonomy
| Hazard | Harm | Description | Considerations | Example | Resources |
| - | - | - | - | - | - |
| Manipulative System Behavior | Dysfunctional Behavior | System optimized for particular types of interaction can lead to dysfunctional behavior. | How might this technology be used to observe patterns of behavior? How could this technology be used to encourage dysfunctional or maladaptive behaviors?	 | A recommendation system optimized for prolonged interaction leads to addiction. |  |
| Manipulative System Behavior | Misinformation | Disguising fake information as legitimate or credible information. | How might this technology be used to generate misinformation? How could it be used to spread misinformation that appears credible? | Generation of synthetic speech of a political leader sways an election. |  |
| Manipulative System Behavior | Distortion of Experienced Reality or Gaslighting | When intentionally misused, technology can undermine trust and distort someone's sense of reality. | Could this be used to modify digital media or physical environments?tions | An IoT device could enable monitoring and controlling of an ex-intimate partner from afar. |  |
| Usage of Public Data | Lack of Informed Consent | Public data is used without informed consent.  | What impact could analyzing or spotlighting data have, even if this data is publicly available? {footcite:p}`boyd2012`  | A deep learning approach is trained on avatars of millions of social media users, who never imagined their avatar would be used in this way {footcite:p}`Raji2020a`  | |

## Economic Well-Being
| Hazard | Harm | Description | Considerations | Example | Resources |
| - | - | - | - | - | - |
| Automation | Devaluation of Individual Expertise and Human Labor | Technology may supplant the use of paid human expertise or labor. | How might this technology impact the need to employ an existing workforce? | AI agents replace radiographers for evaluation of medical imaging. |  |
| Automation | Skill Degradation and Complacency | Over-reliance on automation leads to atrophy of manual skills. | In what ways might this technology reduce the accessibility and ability to use manual controls? | Over-reliance on automation could lead to an inability to gauge the airplane's true orientation because the pilots have been trained to rely on instruments only. |  |

## Dignity
| Hazard | Harm | Description | Considerations | Example | Resources |
| - | - | - | - | - | - |
| Automation | Loss of Human Connection | Removing, reducing, or obscuring visibility of a person's humanity. | How might this technology be used to simplify or abstract the way a person is represented? How might this technology reduce the distinction between humans and the digital world? | Entity recognition and virtual overlays in drone surveillance could reduce the perceived accountability of human actions. |  |
| Human Labor | Exploitation | People might be compelled or misled to work on something that impacts their dignity or well-being. | What role did human labor play in producing training data for this technology? How was this workforce acquired? What role does human labor play in supporting this technology? Where is this workforce expected to come from? | Poorly paid and trained annotators are exploited to label large amount of sensitive and graphic data {footcite:p}`Raji2020a` |  |

```{footbibliography}
```