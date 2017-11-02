<?xml version="1.0" encoding="UTF-8"?>


<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:exist="http://exist.sourceforge.net/NS/exist"
    xmlns="http://www.w3.org/1999/xhtml"
    xmlns:xs="http://www.w3.org/2001/XMLSchema"
    xmlns:tei="http://www.tei-c.org/ns/1.0" 
    xmlns:voice="http://www.univie.ac.at/voice/ns/1.0"
    xmlns:cq="http://www.univie.ac.at/voice/corpusquery"
    version="2.0">


<xsl:output method="text"/>

<xsl:preserve-space elements="tei:u tei:emph"/>


<xsl:template match="tei:TEI">
<xsl:apply-templates select=".//tei:text"/>
</xsl:template>

<xsl:template match="text()[not(ancestor::tei:u)]"/>


<xsl:template match="/tei:TEI/tei:teiHeader"/>

<xsl:template match="tei:vocal">
<xsl:choose>
<xsl:when test="@voice:desc = 'laughing'">
<xsl:if test="@subtype='wordstart'">
<xsl:text> </xsl:text>
</xsl:if>
<xsl:for-each select="1 to @voice:syl">
<xsl:text>@</xsl:text>
</xsl:for-each>
</xsl:when>
</xsl:choose>
</xsl:template>


<xsl:template match="tei:seg[@type = 'overlap']">
<xsl:variable name="num">
<!--   <xsl:call-template name="overlap_number"> -->
<!--     <xsl:with-param name="overlap" select="."/> -->
<!--   </xsl:call-template> -->
</xsl:variable>
<xsl:apply-templates/>
</xsl:template>


<xsl:template match="tei:u">
<xsl:value-of select="replace(@xml:id, '_u_', ':')"/>
<xsl:text>&#x09;</xsl:text>
<xsl:value-of
    select="substring-after(@who,'_')"/><xsl:text>:</xsl:text>
<xsl:text>&#x09;</xsl:text>
<xsl:apply-templates/>
<xsl:text>&#x0D;&#x0A;</xsl:text>
</xsl:template>

<xsl:template match="text()">
  <xsl:value-of select="replace(., '[\n]', ' ')"/>
</xsl:template>

</xsl:stylesheet>
