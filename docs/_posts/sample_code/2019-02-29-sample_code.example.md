---
layout: post
title: sample_code.example
description: >
 Docstring for the example.py module.
---

# sample_code.example
## Description
Docstring for the example.py module.

Modules names should have short, all-lowercase names.  The module name may
have underscores if this improves readability.

Every module should have a docstring at the very top of the file.  The
module's docstring may extend over multiple lines.  If your docstring does
extend over multiple lines, the closing three quotation marks must be on
a line by itself, preferably preceded by a blank line.

---
#### **sample_code.example.foo(** *var1, var2, long_var_name='hi'*  **)** {#signature}

<div class='desc' markdown="1">
Summarize the function in one line.
Several sentences providing an extended description. Refer to
variables using back-ticks, e.g. `var`.
##### Parameters {#section}

<dl>
<dt markdown='1'>`var1` : *array_like*
</dt>
	<dd markdown='1'> Array_like means all those objects -- lists, nested lists, etc. --that can be converted to an array.  We can also refer tovariables like `var1`. 
</dd>

<dt markdown='1'>`var2` : *int*
</dt>
	<dd markdown='1'> The type above can either refer to an actual Python type(e.g. ``int``), or describe the type of the variable in moredetail, e.g. ``(N,) ndarray`` or ``array_like``. 
</dd>

<dt markdown='1'>`long_var_name` : *{'hi', 'ho'}, optional*
</dt>
	<dd markdown='1'> Choices in brackets, default first when optional. 
</dd>

</dl>

##### Returns {#section}

<dl>
<dt markdown='1'>` ` : *type*
</dt>
	<dd markdown='1'> Explanation of anonymous return value of type ``type``. 
</dd>

<dt markdown='1'>`describe` : *type*
</dt>
	<dd markdown='1'> Explanation of return value named `describe`. 
</dd>

<dt markdown='1'>`out` : *type*
</dt>
	<dd markdown='1'> Explanation of `out`. 
</dd>

<dt markdown='1'>` ` : *type_without_description*
</dt>
	<dd markdown='1'>  
</dd>

</dl>

##### Raises {#section}

<dl>
<dt markdown='1'>` ` : *BadException*
</dt>
	<dd markdown='1'> Because you shouldn't have done that. 
</dd>

</dl>

##### Other Parameters {#section}

<dl>
<dt markdown='1'>`only_seldom_used_keywords` : *type*
</dt>
	<dd markdown='1'> Explanation 
</dd>

<dt markdown='1'>`common_parameters_listed_above` : *type*
</dt>
	<dd markdown='1'> Explanation 
</dd>

</dl>

<div class='color-block' markdown='1'>##### **See Also**
numpy.array: Relationship (optional).
numpy.ndarray: Relationship (optional), which could be fairly long, in which case the line wraps here.
numpy.dot, numpy.linalg.norm, numpy.eye: 
</div>
##### Notes {#block-header}
Notes about the implementation algorithm (if needed).

This can have multiple paragraphs.

You may include some math:

$$
 X(e^{j\omega } ) = x(n)e^{ - j\omega n}
$$

And even use a Greek symbol like $$\omega$$ inline.
##### References {#block-header}
Cite the relevant literature, e.g. [1]_.  You may also cite these
references in the notes section above.

.. [1] O. McNoleg, "The integration of GIS, remote sensing,
   expert systems and adaptive co-kriging for environmental habitat
   modelling of the Highland Haggis using object-oriented, fuzzy-logic
   and neural-network techniques," Computers & Geosciences, vol. 22,
   pp. 585-588, 1996.
##### Examples {#block-header}
These are written in doctest format, and should illustrate how to
use the function.

~~~python
>>> a = [1, 2, 3]
>>> print([x + 3 for x in a])
[4, 5, 6]
>>> print("a\nb")
a
b
~~~
---
</div>