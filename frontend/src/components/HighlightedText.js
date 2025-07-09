import React, { useMemo } from 'react';
import { Box, Text, useColorModeValue } from '@chakra-ui/react';
import { motion } from 'framer-motion';

const MotionBox = motion(Box);

const HighlightedText = ({ text, facts, onFactClick }) => {
  const highlightBg = useColorModeValue('rgba(59, 130, 246, 0.1)', 'rgba(59, 130, 246, 0.15)');
  const highlightColor = useColorModeValue('rgb(30, 58, 138)', 'rgb(147, 197, 253)');
  const highlightBorder = useColorModeValue('rgba(59, 130, 246, 0.3)', 'rgba(59, 130, 246, 0.4)');
  
  // Process the text to highlight facts
  const processedContent = useMemo(() => {
    if (!text || !facts || facts.length === 0) {
      return [{ text, isHighlight: false }];
    }
    
    // Sort facts by length (descending) to avoid nested replacements
    const sortedFacts = [...facts].sort((a, b) => b.text.length - a.text.length);
    
    // Create a map of positions to avoid overlapping highlights
    const positions = [];
    
    // Find all occurrences of each fact in the text
    sortedFacts.forEach((fact) => {
      let startIndex = 0;
      while (true) {
        const factIndex = text.indexOf(fact.text, startIndex);
        if (factIndex === -1) break;
        
        // Check if this position overlaps with an existing highlight
        const overlaps = positions.some(
          (pos) => 
            (factIndex >= pos.start && factIndex < pos.end) || 
            (factIndex + fact.text.length > pos.start && factIndex + fact.text.length <= pos.end) ||
            (factIndex <= pos.start && factIndex + fact.text.length >= pos.end)
        );
        
        if (!overlaps) {
          positions.push({
            start: factIndex,
            end: factIndex + fact.text.length,
            fact,
          });
        }
        
        startIndex = factIndex + 1;
      }
    });
    
    // Sort positions by start index
    positions.sort((a, b) => a.start - b.start);
    
    // Build the result array
    const result = [];
    let lastEnd = 0;
    
    positions.forEach((pos) => {
      if (pos.start > lastEnd) {
        result.push({
          text: text.substring(lastEnd, pos.start),
          isHighlight: false,
        });
      }
      
      result.push({
        text: text.substring(pos.start, pos.end),
        isHighlight: true,
        fact: pos.fact,
      });
      
      lastEnd = pos.end;
    });
    
    if (lastEnd < text.length) {
      result.push({
        text: text.substring(lastEnd),
        isHighlight: false,
      });
    }
    
    return result;
  }, [text, facts]);
  
  return (
    <Text whiteSpace="pre-wrap">
      {processedContent.map((item, index) => 
        item.isHighlight ? (
          <MotionBox
            key={index}
            as="span"
            display="inline"
            bg={highlightBg}
            color={highlightColor}
            px={1}
            py={0.5}
            borderRadius="sm"
            borderBottomWidth="1px"
            borderBottomColor={highlightBorder}
            cursor="pointer"
            whiteSpace="normal"
            initial={{ backgroundColor: highlightBg }}
            whileHover={{ 
              backgroundColor: 'rgba(59, 130, 246, 0.15)',
              borderBottomColor: 'rgba(59, 130, 246, 0.5)',
              transition: { duration: 0.15 }
            }}
            onClick={() => onFactClick(item.fact)}
          >
            {item.text}
          </MotionBox>
        ) : (
          <Box key={index} as="span">
            {item.text}
          </Box>
        )
      )}
    </Text>
  );
};

export default HighlightedText; 