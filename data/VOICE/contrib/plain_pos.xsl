<?xml version="1.0" encoding="UTF-8"?>


<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:exist="http://exist.sourceforge.net/NS/exist"
    xmlns="http://www.w3.org/1999/xhtml"
    xmlns:xs="http://www.w3.org/2001/XMLSchema"
    xmlns:tei="http://www.tei-c.org/ns/1.0" 
    xmlns:voice="http://www.univie.ac.at/voice/ns/1.0"
    xmlns:cq="http://www.univie.ac.at/voice/corpusquery"
    version="2.0">


<xsl:key name="FEATSTRUCTS" match="tei:fs" use="@xml:id"/>
<xsl:key name="FEATS" match="tei:f" use="@xml:id"/>

<xsl:output method="text"/>



<xsl:template match="tei:TEI">
<xsl:apply-templates select=".//tei:u"/>
</xsl:template>

<xsl:template match="text()[not(ancestor::tei:w)]"/>

<xsl:template match="tei:u">
  <xsl:variable name="utteranceID">
    <xsl:value-of select="replace(@xml:id, '_u_', '_')"/>
  </xsl:variable>
  <xsl:for-each select="tei:w">
    <xsl:value-of select="string-join(($utteranceID, ':', string(position()), '&#x09;'),'')"/>
    <xsl:apply-templates select="."/>
  </xsl:for-each>
  <xsl:text>&#x0D;&#x0A;</xsl:text>
</xsl:template>


<xsl:template match="tei:w">
  <xsl:value-of select=".//descendant-or-self::tei:w[text() and matches(text(), '\w')]"/>
  <xsl:text>&#x09;</xsl:text>
  <xsl:for-each select="descendant-or-self::tei:w">
    <xsl:if test="@ana and position() gt 1">
      <xsl:text>/</xsl:text>
    </xsl:if>
    <xsl:variable name="fss" select="key('FEATSTRUCTS', substring-after(@ana,'#'))"/>
    <xsl:variable name="fs">
      <xsl:for-each select="tokenize($fss/@feats, ' ')">
        <feat><xsl:value-of select="."/></feat>
      </xsl:for-each>
    </xsl:variable>

    <xsl:for-each select="$fs/*">
      <xsl:variable name="feat" select="substring-after(., '#')"/>
      <xsl:variable name="fName" select="$feat"/>

      <!-- <xsl:variable name="fName"> -->
      <!--   <xsl:value-of select="key('FEATS', $feat)/@name"/> -->
      <!-- </xsl:variable> -->
      <xsl:choose>
        <xsl:when test="starts-with($fName, 'f')">
          <xsl:text>(</xsl:text>
          <xsl:value-of select="substring-after($fName,'f')"/>
          <xsl:text>)</xsl:text>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="$fName"/>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:for-each>
  </xsl:for-each>
  <xsl:text>&#x09;</xsl:text>
  <xsl:value-of select="descendant-or-self::tei:w/@lemma"/>
  <xsl:text>&#x0D;&#x0A;</xsl:text>
</xsl:template>

<xsl:template match="text()">
  <xsl:value-of select="replace(., '[\n]', ' ')"/>
</xsl:template>

</xsl:stylesheet>
