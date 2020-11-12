---
layout: notes
title: software engineering
category: cs
---

#  software engineering

Some notes on software engineering, based on two courses at UVA.

## ch 2 - software engineering
- software engineering - the application of a systematic, disciplined, quantifiable approach to the development, operation, and maintenance of software; that is, the application of engineering to software
- SWEBOK guide
- Polya
	1.	Understand the problem (communication and analysis).
	2.	Plan a solution (modeling and software design).
	3.	Carry out the plan (code generation).
	4.	Examine the result for accuracy (testing and quality assurance).

1. *tools* - provide automated or semi-automated support
2. *methods* - provide technical how-tos
	- ex: communication, requirements analysis, design modeling, program construction, testing and support
3. *process model* - framework for delivering technology
	- framework activities
		- communication, planning, modeling, construction, deployment
	- umbrella activities
		- software project tracking and control, risk management,
software quality assurance, technical reviews, measurement, software configuration management, reusability management, work product preparation and production
4. a *quality focus* - bedrock that supports software engineering
- every software has some business need

## ch 3 - software process structure
- *task set* - defines the actual work to be done to accomplish the objectives of a software engineering action
- *process pattern* - template - describes process-related problem, suggests solutions
	- *stage patterns* - defines problems associated with a framework activity
	- *task patterns*
	- *phase patterns* - define sequence of framework activities
- process assessment and improvement
	- SCAMPI, CBAIPI, SPICE, ISO 9001:2000 for software

## ch 4 - process models
- *prescriptive models* - advocated orderly approach
	- *waterfall model*
		- everything is in order
		- communication, planning, modeling, construction, deployment
	- *V model*
		- move down the model and then test each step on the way up
	- *incremental model*
		- increments (like versions are produced)
- *evolutionary models* - use prototyping, users may think prototype is actual model
	- *spiral* - continually do each step of waterfall
	- *concurrent* - different parts have different interations
- others
	- *component based* - when using outside code
		- the process to apply when reuse is a development objective
	- *formal* - when safety is needed
	- *aspect-oriented* - construct aspects - new paradigm
	- *unified process* - use-case driven, works well with UML
- personal and team software processes

## ch 5 - agile development
- driven by customer descriptions
- adapts as changes occur
- delivers multiple software increments
- self-organizing teams
- at regular intervals, the team reflects on how to become more effective
- *agile unified process* - each AUP iteration addresses several activies
	- modeling, implementation, testing, deployment,...
- *agile modeling* - set of modeling principles
	- model with a purpose
	- use multiple models
	- adapt locally
1. *extreme programming (xp)* - most widely used agile process
	- begins with *user stories*
	- each one gets a cost
	- grouped together for deliverable increment
	- CRC - class responsibility collaborator
	- commitment is made on delivery date
	- project velocity defines subsequent delivery dates
	- encourages unit tests, pair programming
	- acceptance tests defined by the user
2. *industrial xp* - greater inclusion of management, customers
	- readiness assesment, project community, project chartering, test driven management, retrospectives, continuous learning
3. *scrum* - development work is partition into "packets"
	- testing and documentation are on-going
	- work occurs in *sprints* and derived from a *backlog*
	- meetings are short
	- demos are delivered to the customer
4. *dynamic systems development method* - similar to xp

## ch 6 - human aspects
- roles
	- ambassador – represents team to outside constituencies
	- scout – crosses team boundaries to collect information
	- guard – protects access to team work products
	- sentry – controls information sent by stakeholders
	- coordinator – communicates across the team and organization 
- teams
	- sense of purpose, involvement, trust, improvement
	- diversity of skill sets
- *organizational paradigms*
	- closed paradigm — structure by authority
	- random paradigm - loose, depends on individual initiatives
	- open paradigm - mix of closed and random
	- synchronous paradigm - compartmentalizes problem
- some collaboration tools
	- slack
	- gforge
	- onedesk
	- rational team concert
- collaboration, coordination, communication
	
## ch 7 - principles
- process principles
	- be agile, effective team, communicate, assess risk, quality focus, 
- practice principles
	- divide and conquer, transfer of information, abstraction, patterns, maintenance
- communication principles
	- gather customer requirements
- planning principles
	- understand scope, be iterative
- modeling principles
	- *UML diagrams*
	- *requirements models* - *analysis models* - represent customer requirements in information domain, functional domain, behavioral domain
	- *design models* - represent characteristics of software that helps construct it effectively - ex. like an architect's plans for a house
- preparation, construction, testing, deployment
- The deployment activity encompasses 3 actions:
	- Delivery
	- Support
	- Feedback 

## ch 8 - understanding requirements
1. inception - first questions
2. elicitation - requirements understood
3. elaboration 
4. negotiation
5. specification - maybe in models
6. validation - products are assessed
7. requirements managing
- quality function deployment - determines the customer satisfaction of each requirement
- requirement types - functional  describes what software should do, while non-functional place constraints on how
	- Functional Requirement:  A system must send an email whenever a certain condition is met (e.g. an order is placed, a customer registers, etc.
	- Nonfunctional Requirement: Emails should be sent with a latency of no greater than 12 hours from such an activity
- use-cases - collection of user scenarios that describe the thread of usage of a system
	- each described from point-of-view of an actor
- analysis model - provides description for computer-based system
	- scenario-based elements - functional and use-cases
	- class-based elements - implied by scenarios
	- behavioral elements - state diagram
	- flow-oriented elements - data flow diagram
- state diagram - represents the behavior of a system by depicting its states and the events that cause the system to change state
- *UML diagrams*
	- case diagrams
	- activity diagrams
	- class diagrams
	
## ch 9 - requirements modeling  - scenario-based
- analysis modeling - focuses on what not how
- requirements modeling must achieve 3 objectives
	1. to describe what the customer requires
	2. basis for creation of a software design
	3. define a set of requirements that can be validated
- domain analysis - define the domain to be investigated
	- analyze applications in the domain
- scenario-based modeling
	- *use-case* - a scenario that describes a thread of usage
		- *actors* - represent roles people or devices play as the system functions
		- *users* - can play a number of different roles for a given scenario
	- exceptions - situations that cause the system to exhibit unusual behavior
- uml models
	- *activity diagrams* – supplements the use case by providing a graphical representation of the flow of interaction within a specific scenario.
	- *swimlane diagrams* - is a useful variation of the activity diagram and allows you to represent the flow of activities described by the use case at the same time indicate which actor or analysis class has responsibility for the action described by an activity rectangle

## ch 10 - requirements modeling - class-based methods
1. *structured analysis* - considers data and the processes that transform the data as separate entities
2. *object-oriented analysis* - focuses on definition of classes
- class-based modeling - *objects, operations, relationships, collaborations*  
- processing narrative - describes overall description of the function to be developed - nouns underlined
- *attributes* - describe a class that has been selected for inclusion in the analysis model
- operations - define the behavior of an object
- *class-responsibility-collaborator (CRC) modeling* - provides simple means for making classes that are relevant to requirements
	- like index cards with class, responsibilities, collaborators on each card
- class types
	- entity classes - extracted directly from problem statement
	- boundary classes - used to create the interface
	- controller classes - manage a unit of work from start to finish
- collaborations - three different generic relationships
	- associations
		- is-part-of
		- has-knowledge-of
	- dependencies
		- depends-upon
- plus - public visible, minus - private

## ch 11 - requirements modeling - behavior models
- behavioral modeling - depicts the system states and its classes
	- indicates how software will respond to external events or stimuli
- state representations
	- passive state - current status of attributes
	- active state - current status of object as it undergoes continuing transformation
	- event - occurrence that cuases the system to exhibit some behavoir
	- action - process that occurs as a consequence of making a transaction
- semantic analysis pattern - describes set of use cases for application
1. *content Analysis* - The full spectrum of content to be provided by the WebApp is identified,  including text, graphics and images, video, and audio data. Data modeling can be used to identify and describe each of the data objects. 
2. *interaction model*
	- use-cases
	- sequence diagrams
	- state diagrams
	- user interface prototypes
3. *functional model* - describes operations that will be applied to manipulate content and describes other processing functions that are independent of content but necessary to the end user
4. *configuration model* - environment and infrastructure in which the app resides
	- server-side and client-side
5. *navigation modeling* - defines overall navigation strategy for the app